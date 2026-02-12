"""Evaluator agent for analyzing and summarizing discussions."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import BaseAgent, AgentResponse
from ..config import AgentConfig, LLMConfig


@dataclass
class EvaluationResult:
    """Result of discussion evaluation."""
    summary: str
    key_points: List[str]
    perspectives: Dict[str, str]
    consensus_areas: List[str]
    disagreement_areas: List[str]
    quality_score: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "perspectives": self.perspectives,
            "consensus_areas": self.consensus_areas,
            "disagreement_areas": self.disagreement_areas,
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            summary=data["summary"],
            key_points=data["key_points"],
            perspectives=data["perspectives"],
            consensus_areas=data["consensus_areas"],
            disagreement_areas=data["disagreement_areas"],
            quality_score=data["quality_score"],
            recommendations=data["recommendations"],
            metadata=data.get("metadata", {}),
        )


class EvaluatorAgent(BaseAgent):
    """Agent that evaluates and summarizes discussions."""

    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        super().__init__(config, llm_config)

    def generate_response(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate an evaluation response.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            context: Additional context

        Returns:
            AgentResponse with the evaluation
        """
        result = self.evaluate_discussion(topic, conversation_history)
        return AgentResponse(
            agent_name=self.name,
            content=result.summary,
            metadata={"evaluation": result.to_dict()},
        )

    def evaluate_discussion(
        self,
        topic: str,
        conversation_history: List,
    ) -> EvaluationResult:
        """
        Perform a comprehensive evaluation of the discussion.

        Args:
            topic: The discussion topic
            conversation_history: Full conversation history

        Returns:
            EvaluationResult with detailed analysis
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}

全体の議論:
{history_text}

この議論を評価してください。以下のJSON形式で回答してください：

{{
    "summary": "議論全体の要約（200-300文字）",
    "key_points": ["主要な論点1", "主要な論点2", ...],
    "perspectives": {{
        "参加者名1": "その参加者の主な主張の要約",
        "参加者名2": "その参加者の主な主張の要約"
    }},
    "consensus_areas": ["合意が得られた点1", "合意が得られた点2", ...],
    "disagreement_areas": ["意見が分かれた点1", "意見が分かれた点2", ...],
    "quality_score": 0.0から1.0の議論の質スコア,
    "recommendations": ["今後の議論への提言1", "今後の議論への提言2", ...]
}}

JSONのみを出力してください。
"""
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, self.get_system_prompt())

        # Parse JSON response
        import json
        try:
            # Clean up response if needed
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)
            return EvaluationResult(
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                perspectives=data.get("perspectives", {}),
                consensus_areas=data.get("consensus_areas", []),
                disagreement_areas=data.get("disagreement_areas", []),
                quality_score=float(data.get("quality_score", 0.5)),
                recommendations=data.get("recommendations", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to simple summary
            return EvaluationResult(
                summary=response,
                key_points=[],
                perspectives={},
                consensus_areas=[],
                disagreement_areas=[],
                quality_score=0.5,
                recommendations=[],
                metadata={"parse_error": str(e)},
            )

    def generate_interim_assessment(
        self,
        topic: str,
        conversation_history: List,
        round_number: int,
    ) -> AgentResponse:
        """
        Generate an interim assessment during the discussion.

        Args:
            topic: The discussion topic
            conversation_history: List of messages so far
            round_number: Current round number

        Returns:
            AgentResponse with interim assessment
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

中間評価として以下を簡潔に述べてください：
1. これまでの主な論点
2. 議論の進行状況
3. 未解決の問題点
4. 次のラウンドへの提案

評価者として客観的な視点から分析してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "interim_assessment", "round": round_number},
        )

    def generate_quality_feedback(
        self,
        topic: str,
        conversation_history: List,
    ) -> AgentResponse:
        """
        Generate feedback on discussion quality.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages

        Returns:
            AgentResponse with quality feedback
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-10:]  # Recent messages
        ])

        prompt = f"""
議論のトピック: {topic}

最近の議論:
{history_text}

議論の質について以下の観点からフィードバックしてください：
1. 論理性: 主張に根拠があるか
2. 建設性: 議論が前に進んでいるか
3. 多様性: 様々な視点が出ているか
4. 敬意: 参加者間で敬意が保たれているか

改善点があれば具体的に指摘してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "quality_feedback"},
        )

    def identify_logical_fallacies(
        self,
        topic: str,
        conversation_history: List,
    ) -> AgentResponse:
        """
        Identify any logical fallacies in the discussion.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages

        Returns:
            AgentResponse with identified fallacies
        """
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ])

        prompt = f"""
議論のトピック: {topic}

議論の内容:
{history_text}

この議論の中で見られる論理的誤謬（ロジカルファラシー）を特定してください。
以下の形式で回答してください：

1. [発言者名] - [誤謬の種類]
   該当する発言: 「...」
   説明: なぜこれが論理的誤謬か

誤謬が見つからない場合は、「特に顕著な論理的誤謬は見つかりませんでした」と回答してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={"type": "fallacy_analysis"},
        )
