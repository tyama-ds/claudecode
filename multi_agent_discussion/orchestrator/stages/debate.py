"""Debate stage - agents discuss based on their research findings."""

from typing import Optional

from .base import BaseStage
from ..context import PipelineContext


class DebateStage(BaseStage):
    """
    Conducts a multi-agent debate based on research findings.

    Each agent argues from their research perspective. The debate
    is moderated and evaluated, producing a discussion transcript
    and evaluation summary.
    """

    stage_type = "debate"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.research_results:
            raise ValueError("No research results available for debate.")

        self._report_progress("議論を準備中...", 0.1)

        from ...config import (
            Config as DiscussionConfig,
            LLMConfig,
            AgentConfig,
            DiscussionConfig as DiscConfig,
            AgentRole,
        )
        from ...main import MultiAgentDiscussion

        debate_config = self.config.debate
        llm_config = self.config.llm_config

        # Build agents from research results
        agents = []

        # Moderator
        agents.append(AgentConfig(
            name="モデレーター",
            role=AgentRole.MODERATOR,
            llm_config=llm_config,
        ))

        # Participants based on research agents
        for name, result in context.research_results.items():
            # Inject research findings into persona
            persona = (
                f"あなたは「{result.perspective}」の視点から調査を行いました。\n"
                f"調査結果の要約:\n{result.report_content[:2000]}"
            )
            agent_cfg = AgentConfig(
                name=name,
                role=AgentRole.PARTICIPANT,
                persona=persona,
                llm_config=llm_config,
            )
            agents.append(agent_cfg)

        # Evaluator
        agents.append(AgentConfig(
            name="評価者",
            role=AgentRole.EVALUATOR,
            llm_config=llm_config,
        ))

        # Create discussion config
        discussion_cfg = DiscussionConfig(
            llm=llm_config,
            discussion=DiscConfig(
                topic=context.topic,
                max_rounds=debate_config.max_rounds,
                enable_evaluation=True,
                save_session=False,
            ),
            agents=agents,
        )

        self._report_progress("議論を実行中...", 0.3)

        discussion = MultiAgentDiscussion(discussion_cfg)
        result = discussion.run()

        context.discussion_transcript = result.get("transcript", "")
        context.discussion_evaluation = result.get("evaluation", {})

        self._report_progress("議論完了", 1.0)
        return context
