"""Participant agent for contributing to discussions."""

from typing import List, Optional

from .base import BaseAgent, AgentResponse
from ..config import AgentConfig, LLMConfig


class ParticipantAgent(BaseAgent):
    """Agent that participates in the discussion with a specific persona."""

    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        super().__init__(config, llm_config)
        self.persona = config.persona

    def get_system_prompt(self) -> str:
        """Get the system prompt including persona."""
        base = super().get_system_prompt()
        return f"{base}\n\nあなたの名前は「{self.name}」です。この名前で議論に参加してください。"

    def generate_response(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a participant response to the discussion.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            context: Additional context or question from moderator

        Returns:
            AgentResponse with the participant's contribution
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}

これまでの議論:
{history_text}

あなたの立場・ペルソナ: {self.persona}

上記の議論を踏まえて、あなたの視点から意見を述べてください。
以下の点に注意してください：
1. 他の参加者の意見に言及しながら自分の考えを述べる
2. 具体的な根拠や例を挙げる
3. 建設的な議論に貢献する
4. 自分の立場を明確にしつつも、他の意見も尊重する

回答は簡潔に、議論の参加者として自然な発言形式で記述してください。
"""
        if context:
            prompt += f"\n\nモデレーターからの質問/指示: {context}"

        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "contribution",
                "persona": self.persona,
            },
        )

    def generate_initial_opinion(
        self,
        topic: str,
        opening_statement: str,
    ) -> AgentResponse:
        """
        Generate an initial opinion at the start of discussion.

        Args:
            topic: The discussion topic
            opening_statement: The moderator's opening statement

        Returns:
            AgentResponse with the initial opinion
        """
        prompt = f"""
議論のトピック: {topic}

モデレーターの開会宣言:
{opening_statement}

あなたの立場・ペルソナ: {self.persona}

議論の最初の発言者として、トピックに対するあなたの基本的な立場と考えを述べてください。
- 明確な立場を示す
- その立場の根拠を簡潔に説明する
- 他の参加者が反応しやすい形で意見を述べる

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "initial_opinion",
                "persona": self.persona,
            },
        )

    def generate_rebuttal(
        self,
        topic: str,
        conversation_history: List,
        target_agent: str,
        target_argument: str,
    ) -> AgentResponse:
        """
        Generate a rebuttal to a specific argument.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            target_agent: Name of the agent to rebut
            target_argument: The argument to rebut

        Returns:
            AgentResponse with the rebuttal
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-5:]  # Recent context
        ])

        prompt = f"""
議論のトピック: {topic}

最近の議論:
{history_text}

あなたの立場・ペルソナ: {self.persona}

{target_agent}の以下の主張に対して反論してください：
「{target_argument}」

反論は以下の点に注意してください：
1. 相手の主張を正確に理解した上で反論する
2. 感情的にならず、論理的に反論する
3. 具体的な反例や根拠を示す
4. 相手の人格ではなく主張に対して反論する

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "rebuttal",
                "target_agent": target_agent,
                "persona": self.persona,
            },
        )

    def generate_agreement(
        self,
        topic: str,
        conversation_history: List,
        target_agent: str,
        target_argument: str,
    ) -> AgentResponse:
        """
        Generate an agreement with additional perspective.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            target_agent: Name of the agent to agree with
            target_argument: The argument to agree with

        Returns:
            AgentResponse with the agreement and additional perspective
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-5:]
        ])

        prompt = f"""
議論のトピック: {topic}

最近の議論:
{history_text}

あなたの立場・ペルソナ: {self.persona}

{target_agent}の以下の主張に同意し、さらに発展させてください：
「{target_argument}」

回答は以下の点に注意してください：
1. なぜ同意するのか簡潔に説明
2. 自分の視点から補足や発展を加える
3. 新たな視点や具体例を提供する

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "agreement",
                "target_agent": target_agent,
                "persona": self.persona,
            },
        )
