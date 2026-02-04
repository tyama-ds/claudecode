"""Moderator agent for managing discussion flow."""

from typing import List, Optional

from .base import BaseAgent, AgentResponse
from ..config import AgentConfig, LLMConfig


class ModeratorAgent(BaseAgent):
    """Agent that moderates the discussion."""

    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        super().__init__(config, llm_config)

    def generate_opening(self, topic: str) -> AgentResponse:
        """
        Generate an opening statement for the discussion.

        Args:
            topic: The discussion topic

        Returns:
            AgentResponse with the opening statement
        """
        prompt = f"""
議論のトピック: {topic}

議論の開始にあたり、以下を行ってください：
1. トピックの簡潔な紹介
2. 議論の目的の説明
3. 参加者への最初の問いかけ

回答は自然な議論の進行役として、参加者に向けて話しかける形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "opening"},
        )

    def generate_response(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a moderator response to guide the discussion.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            context: Additional context

        Returns:
            AgentResponse with the moderator's guidance
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}

これまでの議論:
{history_text}

モデレーターとして、以下のいずれかを行ってください：
1. 議論を次の論点に進める
2. 特定の参加者に深掘りを求める
3. 異なる視点を促す
4. 論点を整理する

簡潔に、議論を建設的に進める発言をしてください。
"""
        if context:
            prompt += f"\n追加の指示: {context}"

        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "moderation"},
        )

    def generate_round_summary(
        self,
        topic: str,
        conversation_history: List,
        round_number: int,
    ) -> AgentResponse:
        """
        Generate a summary of the current round.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            round_number: Current round number

        Returns:
            AgentResponse with the round summary
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}
現在のラウンド: {round_number}

これまでの議論:
{history_text}

このラウンドで出た意見を簡潔にまとめ、次のラウンドへの導入を行ってください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "round_summary", "round": round_number},
        )

    def generate_closing(
        self,
        topic: str,
        conversation_history: List,
    ) -> AgentResponse:
        """
        Generate a closing statement for the discussion.

        Args:
            topic: The discussion topic
            conversation_history: Full conversation history

        Returns:
            AgentResponse with the closing statement
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}

全体の議論:
{history_text}

議論を締めくくるにあたり：
1. 主要な論点のまとめ
2. 各立場の要約
3. 締めの言葉

を簡潔に述べてください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "closing"},
        )

    def should_continue_discussion(
        self,
        topic: str,
        conversation_history: List,
        current_round: int,
        max_rounds: int,
    ) -> bool:
        """
        Determine if the discussion should continue.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            current_round: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if discussion should continue, False otherwise
        """
        if current_round >= max_rounds:
            return False

        if current_round < 2:  # Minimum 2 rounds
            return True

        # Ask LLM to evaluate if discussion should continue
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-10:]  # Last 10 messages
        ])

        prompt = f"""
議論のトピック: {topic}
現在のラウンド: {current_round}/{max_rounds}

最近の議論:
{history_text}

この議論は続けるべきですか？以下の観点から判断してください：
1. 新しい論点が出ているか
2. 議論が収束に向かっているか
3. 参加者間で建設的な対話が続いているか

回答は「続ける」または「終了」の一言のみでお願いします。
"""
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, self.get_system_prompt())

        return "続ける" in response or "継続" in response
