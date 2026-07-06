"""Report generation stage - produces the final output."""

from datetime import datetime
from pathlib import Path

from .base import BaseStage
from ..context import PipelineContext


class ReportStage(BaseStage):
    """
    Generates the final report from all pipeline data.

    Compiles research results, discussion transcripts, fact-check results,
    and synthesized content into a single comprehensive report.
    """

    stage_type = "report"

    def execute(self, context: PipelineContext) -> PipelineContext:
        report_config = self.config.report
        llm_config = self.config.get_llm_config(report_config.llm_config)

        self._report_progress("最終レポート生成中...", 0.2)

        # Build the final report
        sections = []

        # Title
        sections.append(f"# {context.topic}\n")
        sections.append(f"*生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        sections.append(f"*パイプラインID: {context.pipeline_id}*\n")

        # Executive summary using LLM
        self._report_progress("エグゼクティブサマリー生成中...", 0.3)
        summary = self._generate_executive_summary(context, llm_config)
        sections.append(f"## エグゼクティブサマリー\n\n{summary}\n")

        # Main content (synthesized/refined/best report)
        main_content = context.get_latest_report()
        if main_content:
            sections.append(f"## 調査・分析結果\n\n{main_content}\n")

        # Individual research summaries (if configured)
        if report_config.include_sources and context.research_results:
            sections.append("## 各視点からの調査要約\n")
            for name, result in context.research_results.items():
                sections.append(f"### {name}（{result.perspective}）\n")
                # Truncate long reports
                content = result.report_content
                if len(content) > 2000:
                    content = content[:2000] + "\n\n*（以下省略）*"
                sections.append(f"{content}\n")

        # Discussion transcript
        if report_config.include_discussion and context.discussion_transcript:
            sections.append(f"## 議論の記録\n\n{context.discussion_transcript}\n")

        # Discussion evaluation
        if context.discussion_evaluation:
            eval_data = context.discussion_evaluation
            sections.append("## 議論の評価\n")
            if isinstance(eval_data, dict):
                if eval_data.get("summary"):
                    sections.append(f"{eval_data['summary']}\n")
                if eval_data.get("key_points"):
                    sections.append("**主要論点:**\n")
                    for point in eval_data["key_points"]:
                        sections.append(f"- {point}")
                    sections.append("")

        # Competitive rankings
        if context.competitive_rankings:
            sections.append("## 競争評価結果\n")
            for rank in sorted(context.competitive_rankings, key=lambda r: r.get("total_score", 0), reverse=True):
                score = rank.get("total_score", 0)
                name = rank.get("agent_name", "")
                sections.append(f"- **{name}**: スコア {score:.2f}")
                if rank.get("strengths"):
                    sections.append(f"  - 強み: {', '.join(rank['strengths'])}")
            sections.append("")

        # Fact-check results
        if report_config.include_fact_check and context.fact_check_results:
            fc = context.fact_check_results
            sections.append("## ファクトチェック結果\n")
            score = fc.get("accuracy_score", "N/A")
            sections.append(f"**信頼性スコア:** {score}\n")

            if fc.get("verification_text"):
                sections.append(f"{fc['verification_text']}\n")
            elif fc.get("results"):
                for r in fc["results"][:10]:
                    verdict = r.get("verdict", "")
                    claim = r.get("claim", "")
                    sections.append(f"- [{verdict}] {claim}")
                sections.append("")

        # Pipeline stage history
        sections.append("## パイプライン実行履歴\n")
        for stage_result in context.stage_results:
            status = stage_result.metadata.get("status", "unknown")
            sections.append(f"- **{stage_result.stage_name}** ({stage_result.stage_type}): {status}")
        sections.append("")

        # Compile final report
        final_report = "\n".join(sections)
        context.final_report = final_report

        # Save to file
        self._report_progress("レポートを保存中...", 0.8)
        output_dir = Path(report_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ext_map = {
            "markdown": ".md",
            "html": ".html",
            "docx": ".docx",
            "pdf": ".pdf",
        }
        ext = ext_map.get(report_config.output_format, ".md")
        output_path = output_dir / f"report_{context.pipeline_id}{ext}"

        if report_config.output_format == "markdown":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_report)
        elif report_config.output_format == "html":
            html = self._markdown_to_html(final_report)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            # For docx/pdf, save as markdown and note format
            md_path = output_dir / f"report_{context.pipeline_id}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            output_path = md_path

        context.final_report_path = str(output_path)
        self._report_progress("レポート生成完了", 1.0)
        return context

    def _generate_executive_summary(self, context: PipelineContext, llm_config) -> str:
        """Generate an executive summary from all context data."""
        # Gather all available content
        content_parts = []
        if context.synthesized_report:
            content_parts.append(f"統合レポート:\n{context.synthesized_report[:2000]}")
        elif context.best_report:
            content_parts.append(f"最優秀レポート:\n{context.best_report[:2000]}")
        else:
            for name, result in context.research_results.items():
                content_parts.append(f"{name}:\n{result.report_content[:500]}")

        if context.discussion_evaluation and isinstance(context.discussion_evaluation, dict):
            summary = context.discussion_evaluation.get("summary", "")
            if summary:
                content_parts.append(f"議論の評価:\n{summary}")

        all_content = "\n\n".join(content_parts)

        prompt = f"""
以下のコンテンツに基づいて、「{context.topic}」のエグゼクティブサマリーを200〜300文字で作成してください。

{all_content}

重要な発見、結論、提言を簡潔にまとめてください。
"""
        agent = self._create_temp_agent(llm_config)
        return agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは簡潔で的確なエグゼクティブサマリーを作成する専門家です。",
        )

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML."""
        try:
            import markdown
            html_body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
        except ImportError:
            # Simple fallback
            html_body = markdown_text.replace("\n", "<br>\n")

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 2em; line-height: 1.6; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    def _create_temp_agent(self, llm_config):
        from ...agents.base import BaseAgent
        from ...config import AgentConfig, AgentRole

        temp_config = AgentConfig(name="reporter", role=AgentRole.EVALUATOR)

        class _TempAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        return _TempAgent(temp_config, llm_config)
