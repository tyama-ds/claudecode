"""Parallel research stage using deep_research_tool."""

import json
from pathlib import Path
from typing import Optional

from .base import BaseStage
from ..context import PipelineContext, ResearchResult
from ..config import OrchestratorConfig


class ParallelResearchStage(BaseStage):
    """
    Conducts parallel research from multiple perspectives.

    Each research agent investigates the topic from its own angle using
    deep_research_tool, or loads results from existing session files.
    """

    stage_type = "parallel_research"

    def execute(self, context: PipelineContext) -> PipelineContext:
        agents = self.config.research_agents
        total = len(agents)

        for i, agent_config in enumerate(agents):
            progress = i / total
            self._report_progress(f"調査中: {agent_config.name}", progress)

            if agent_config.from_file:
                result = self._load_from_file(agent_config)
            else:
                result = self._run_research(agent_config, context.topic)

            context.add_research_result(result)

        self._report_progress(f"{total}件の調査完了", 1.0)
        return context

    def _load_from_file(self, agent_config) -> ResearchResult:
        """Load research result from an existing session file."""
        filepath = Path(agent_config.from_file)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract content from deep_research_tool session format
        report_content = ""
        evidence = []

        if "section_contents" in data:
            # deep_research_tool session format
            for section_id, section in data.get("section_contents", {}).items():
                report_content += f"## {section.get('title', section_id)}\n\n"
                report_content += section.get("content", "") + "\n\n"
                for src in section.get("sources", []):
                    evidence.append({"url": src, "section": section_id})
        elif "report_content" in data:
            # Our own format
            report_content = data["report_content"]
            evidence = data.get("evidence", [])
        else:
            # Plain text / markdown
            report_content = json.dumps(data, ensure_ascii=False, indent=2)

        return ResearchResult(
            agent_name=agent_config.name,
            perspective=agent_config.perspective,
            report_content=report_content,
            evidence=evidence,
            session_path=str(filepath),
            metadata={"source": "file"},
        )

    def _run_research(self, agent_config, topic: str) -> ResearchResult:
        """Run deep_research_tool for an agent."""
        # Build query incorporating perspective
        query = agent_config.research_query or (
            f"{topic} - {agent_config.perspective}"
        )

        llm_config = agent_config.llm_config or self.config.llm_config
        provider = llm_config.provider.value
        model = llm_config.get_model()
        proxy_url = llm_config.proxy_url

        try:
            from deep_research_tool import DeepResearchTool
            from deep_research_tool.config import create_config as create_drt_config

            drt_config = create_drt_config(
                provider=provider,
                model=model,
            )
            # Apply proxy if set
            if proxy_url:
                drt_config.api.proxy_url = getattr(drt_config.api, "proxy_url", proxy_url)

            drt_config.research.max_iterations = agent_config.max_iterations
            drt_config.search.method = agent_config.search_method

            tool = DeepResearchTool(drt_config)
            result = tool.run(
                query=query,
                requirements=f"調査の視点: {agent_config.perspective}",
            )

            # Extract content from session
            session = result.get("session")
            report_content = ""
            evidence = []

            if session and hasattr(session, "section_contents"):
                for section_id, section in session.section_contents.items():
                    report_content += f"## {section.get('title', section_id)}\n\n"
                    report_content += section.get("content", "") + "\n\n"
                    for src in section.get("sources", []):
                        evidence.append({"url": src, "section": section_id})
            elif "report_path" in result:
                report_path = Path(result["report_path"])
                if report_path.exists():
                    report_content = report_path.read_text(encoding="utf-8")

            return ResearchResult(
                agent_name=agent_config.name,
                perspective=agent_config.perspective,
                report_content=report_content,
                evidence=evidence,
                session_id=result.get("session_id", ""),
                session_path=result.get("session_path"),
                metadata={"source": "deep_research_tool", "provider": provider},
            )

        except ImportError:
            # Fallback: use LLM directly for basic research
            return self._fallback_research(agent_config, topic, llm_config)

    def _fallback_research(self, agent_config, topic, llm_config) -> ResearchResult:
        """Fallback research using LLM directly when deep_research_tool unavailable."""
        from ...agents.base import BaseAgent

        # Create a temporary agent for LLM access
        from ...config import AgentConfig, AgentRole
        temp_config = AgentConfig(
            name=agent_config.name,
            role=AgentRole.PARTICIPANT,
            persona=agent_config.perspective,
        )

        class _TempAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        agent = _TempAgent(temp_config, llm_config)

        prompt = f"""
以下のトピックについて、指定された視点から詳細な調査レポートを作成してください。

トピック: {topic}
調査の視点: {agent_config.perspective}

以下の構成でレポートを作成してください：
1. 概要
2. 主要な発見事項
3. 詳細分析
4. 結論と提言

レポートはMarkdown形式で記述してください。
"""
        content = agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt=f"あなたは「{agent_config.perspective}」の専門家リサーチャーです。",
        )

        return ResearchResult(
            agent_name=agent_config.name,
            perspective=agent_config.perspective,
            report_content=content,
            evidence=[],
            metadata={"source": "llm_fallback", "provider": llm_config.provider.value},
        )
