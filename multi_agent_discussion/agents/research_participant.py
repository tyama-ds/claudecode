"""Research-capable participant agent for discussions with information gathering."""

from typing import List, Optional

from .base import AgentResponse
from .search_mixin import SearchCapabilityMixin, AgentSearchConfig
from .participant import ParticipantAgent


class ResearchParticipantAgent(SearchCapabilityMixin, ParticipantAgent):
    """
    Participant agent that searches for information before responding.

    This agent performs web searches based on its persona and the discussion
    context, then uses the search results to inform its contributions.

    MRO: SearchCapabilityMixin -> ParticipantAgent -> BaseAgent
    SearchCapabilityMixin.__init__ calls super().__init__ which chains to ParticipantAgent.
    """

    def __init__(
        self,
        config,
        llm_config,
        search_config: Optional[AgentSearchConfig] = None,
    ):
        """
        Initialize the research participant agent.

        Args:
            config: AgentConfig for the agent
            llm_config: LLMConfig for LLM access
            search_config: Optional search configuration
        """
        # SearchCapabilityMixin.__init__ will call ParticipantAgent.__init__
        super().__init__(config=config, llm_config=llm_config, search_config=search_config)

    def generate_response(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a response with search-backed information.

        Flow:
        1. Generate search queries based on persona + discussion context
        2. Execute searches
        3. Build enhanced prompt with search results
        4. Generate response

        Args:
            topic: Discussion topic
            conversation_history: List of Message objects
            context: Optional additional context (e.g., moderator's question)

        Returns:
            AgentResponse with search metadata
        """
        # Phase 1: Research
        search_context, search_metadata = self.research_and_build_context(
            topic, conversation_history, context
        )

        # Phase 2: Build enhanced prompt
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history
        ]) if conversation_history else "(これまでの議論なし)"

        prompt = f"""議論のトピック: {topic}

これまでの議論:
{history_text}

あなたの立場・ペルソナ: {self.persona}
"""
        if search_context:
            prompt += f"""
あなたが収集した最新の情報:
{search_context}

"""
        prompt += """上記の議論と収集した情報を踏まえて、あなたの視点から意見を述べてください。

以下の点に注意してください：
1. 収集した情報に基づいて具体的な根拠やデータを引用する
2. 他の参加者の意見に言及しながら自分の考えを述べる
3. 情報源を明示する（「〜によると」「〜のデータでは」など）
4. 建設的な議論に貢献する
5. 自分の立場を明確にしつつも、他の意見も尊重する

回答は簡潔に、議論の参加者として自然な発言形式で記述してください。
"""
        if context:
            prompt += f"\nモデレーターからの質問/指示: {context}"

        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "research_contribution",
                "persona": self.persona,
                "search_results": search_metadata,
            },
        )

    def generate_initial_opinion(
        self,
        topic: str,
        opening_statement: str,
    ) -> AgentResponse:
        """
        Generate an initial opinion with search-backed information.

        Args:
            topic: Discussion topic
            opening_statement: Moderator's opening statement

        Returns:
            AgentResponse with search metadata
        """
        # Research before initial opinion (no conversation history yet)
        search_context, search_metadata = self.research_and_build_context(
            topic, [], None
        )

        prompt = f"""議論のトピック: {topic}

モデレーターの開会宣言:
{opening_statement}

あなたの立場・ペルソナ: {self.persona}
"""
        if search_context:
            prompt += f"""
あなたが事前に収集した情報:
{search_context}

"""
        prompt += """議論の最初の発言者として、トピックに対するあなたの基本的な立場と考えを述べてください。

以下の点に注意してください：
1. 収集した情報に基づいて明確な立場を示す
2. その立場の根拠を具体的なデータや情報源を引用して説明する
3. 他の参加者が反応しやすい形で意見を述べる
4. あなたの専門分野の視点から価値のある情報を提供する

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "research_initial_opinion",
                "persona": self.persona,
                "search_results": search_metadata,
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
        Generate a research-backed rebuttal to another agent's argument.

        Args:
            topic: Discussion topic
            conversation_history: List of Message objects
            target_agent: Name of the agent being rebutted
            target_argument: The argument being rebutted

        Returns:
            AgentResponse with search metadata
        """
        # Research to find counter-evidence
        rebuttal_context = f"「{target_agent}」の主張「{target_argument[:100]}...」に対する反論の根拠"
        search_context, search_metadata = self.research_and_build_context(
            topic, conversation_history[-5:], rebuttal_context
        )

        recent_history = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-5:]
        ]) if conversation_history else ""

        prompt = f"""議論のトピック: {topic}

最近の議論:
{recent_history}

あなたの立場・ペルソナ: {self.persona}

反論対象:
{target_agent}の主張: {target_argument}
"""
        if search_context:
            prompt += f"""
反論のために収集した情報:
{search_context}

"""
        prompt += """上記の主張に対して、あなたの視点から反論を述べてください。

以下の点に注意してください：
1. 人ではなく、主張に対して論理的に反論する
2. 収集した情報やデータを引用して具体的な反論を行う
3. 建設的な批判を心がける
4. 自分の代替案や視点も提示する

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "research_rebuttal",
                "persona": self.persona,
                "target_agent": target_agent,
                "search_results": search_metadata,
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
        Generate a research-backed agreement and extension of another's argument.

        Args:
            topic: Discussion topic
            conversation_history: List of Message objects
            target_agent: Name of the agent being agreed with
            target_argument: The argument being supported

        Returns:
            AgentResponse with search metadata
        """
        # Research to find supporting evidence
        agreement_context = f"「{target_agent}」の主張「{target_argument[:100]}...」を支持する追加の根拠"
        search_context, search_metadata = self.research_and_build_context(
            topic, conversation_history[-5:], agreement_context
        )

        recent_history = "\n".join([
            f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-5:]
        ]) if conversation_history else ""

        prompt = f"""議論のトピック: {topic}

最近の議論:
{recent_history}

あなたの立場・ペルソナ: {self.persona}

賛同対象:
{target_agent}の主張: {target_argument}
"""
        if search_context:
            prompt += f"""
賛同を裏付ける追加情報:
{search_context}

"""
        prompt += """上記の主張に賛同し、さらに発展させてください。

以下の点に注意してください：
1. なぜその主張に賛同するか簡潔に説明する
2. 収集した情報から追加の根拠やデータを提供する
3. あなたの専門分野の視点から主張をさらに発展させる
4. 建設的な議論の深化に貢献する

回答は議論の参加者として自然な発言形式で記述してください。
"""
        messages = [{"role": "user", "content": prompt}]
        content = self._call_llm(messages, self.get_system_prompt())

        return AgentResponse(
            agent_name=self.name,
            content=content,
            metadata={
                "type": "research_agreement",
                "persona": self.persona,
                "target_agent": target_agent,
                "search_results": search_metadata,
            },
        )
