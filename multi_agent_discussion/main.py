"""Main orchestrator for multi-agent discussion."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from .config import (
    Config,
    DiscussionState,
    AgentRole,
    create_config,
)
from .agents import (
    create_agent,
    ModeratorAgent,
    ParticipantAgent,
    EvaluatorAgent,
    EvaluationResult,
)
from .conversation import (
    DiscussionSession,
    Message,
    MessageType,
)


class MultiAgentDiscussion:
    """
    Main orchestrator for multi-agent discussions.

    This class manages the entire discussion flow, coordinating between
    moderator, participants, and evaluator agents.
    """

    def __init__(self, config: Config):
        """
        Initialize the discussion orchestrator.

        Args:
            config: Configuration for the discussion
        """
        self.config = config
        self._validate_config()

        # Initialize agents
        self.moderator: Optional[ModeratorAgent] = None
        self.participants: List[ParticipantAgent] = []
        self.evaluator: Optional[EvaluatorAgent] = None
        self._initialize_agents()

        # Session management
        self.session: Optional[DiscussionSession] = None

        # Callbacks
        self._progress_callback: Optional[Callable[[str, float], None]] = None
        self._message_callback: Optional[Callable[[Message], None]] = None

    def _validate_config(self):
        """Validate the configuration."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def _initialize_agents(self):
        """Initialize all agents from configuration."""
        for agent_config in self.config.agents:
            # Use agent-specific LLM config or fall back to global
            llm_config = agent_config.llm_config or self.config.llm

            agent = create_agent(agent_config, llm_config)

            if agent_config.role == AgentRole.MODERATOR:
                self.moderator = agent
            elif agent_config.role in (AgentRole.PARTICIPANT, AgentRole.RESEARCH_PARTICIPANT):
                self.participants.append(agent)
            elif agent_config.role == AgentRole.EVALUATOR:
                self.evaluator = agent

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def set_message_callback(self, callback: Callable[[Message], None]):
        """Set callback for new messages."""
        self._message_callback = callback

    def _report_progress(self, status: str, progress: float):
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(status, progress)

    def _add_message(
        self,
        agent_name: str,
        content: str,
        message_type: MessageType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the session and trigger callback."""
        message = self.session.add_message(
            agent_name=agent_name,
            content=content,
            message_type=message_type,
            metadata=metadata,
        )
        if self._message_callback:
            self._message_callback(message)
        return message

    def run(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        message_callback: Optional[Callable[[Message], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete discussion.

        Args:
            progress_callback: Optional callback for progress updates
            message_callback: Optional callback for new messages

        Returns:
            Dictionary with discussion results
        """
        if progress_callback:
            self.set_progress_callback(progress_callback)
        if message_callback:
            self.set_message_callback(message_callback)

        # Initialize session
        self.session = DiscussionSession(
            topic=self.config.discussion.topic,
            participants=[a.name for a in self.participants] + [self.moderator.name],
        )

        try:
            # Phase 1: Opening
            self._report_progress("議論を開始しています...", 0.0)
            self._run_opening_phase()

            # Phase 2: Main Discussion
            self._report_progress("議論を進行中...", 0.1)
            self._run_discussion_phase()

            # Phase 3: Closing
            self._report_progress("議論をまとめています...", 0.8)
            self._run_closing_phase()

            # Phase 4: Evaluation (optional)
            evaluation_result = None
            if self.config.discussion.enable_evaluation and self.evaluator:
                self._report_progress("議論を評価しています...", 0.9)
                evaluation_result = self._run_evaluation_phase()

            # Save session
            session_path = None
            if self.config.discussion.save_session:
                session_path = self.session.save(
                    session_dir=self.config.discussion.session_dir
                )

            self._report_progress("完了", 1.0)

            return {
                "session_id": self.session.session_id,
                "topic": self.session.topic,
                "rounds": len(self.session.rounds),
                "message_count": self.session.message_count,
                "transcript": self.session.generate_transcript(),
                "evaluation": evaluation_result.to_dict() if evaluation_result else None,
                "session_path": str(session_path) if session_path else None,
            }

        except Exception as e:
            # Save session even on error
            if self.config.discussion.save_session:
                self.session.metadata["error"] = str(e)
                self.session.save(session_dir=self.config.discussion.session_dir)
            raise

    def _run_opening_phase(self):
        """Run the opening phase of the discussion."""
        self.session.transition_state(DiscussionState.OPENING)
        self.session.start_round()

        # Moderator opens the discussion
        self.session.start_turn(self.moderator.name)
        opening = self.moderator.generate_opening(self.config.discussion.topic)
        self._add_message(
            agent_name=opening.agent_name,
            content=opening.content,
            message_type=MessageType.OPENING,
            metadata=opening.metadata,
        )
        self.session.complete_turn()

        # Each participant gives initial opinion
        for participant in self.participants:
            self.session.start_turn(participant.name)
            response = participant.generate_initial_opinion(
                topic=self.config.discussion.topic,
                opening_statement=opening.content,
            )
            self._add_message(
                agent_name=response.agent_name,
                content=response.content,
                message_type=MessageType.CONTRIBUTION,
                metadata=response.metadata,
            )
            self.session.complete_turn()

        self.session.complete_round("オープニングラウンド完了")
        self.session.transition_state(DiscussionState.DISCUSSING)

    def _run_discussion_phase(self):
        """Run the main discussion phase."""
        max_rounds = self.config.discussion.max_rounds
        current_round = 1

        while current_round <= max_rounds:
            progress = 0.1 + (0.7 * current_round / max_rounds)
            self._report_progress(f"ラウンド {current_round}/{max_rounds}", progress)

            self.session.start_round()

            # Moderator guides the discussion
            self.session.start_turn(self.moderator.name)
            moderation = self.moderator.generate_response(
                topic=self.config.discussion.topic,
                conversation_history=self.session.all_messages,
            )
            self._add_message(
                agent_name=moderation.agent_name,
                content=moderation.content,
                message_type=MessageType.MODERATION,
                metadata=moderation.metadata,
            )
            self.session.complete_turn()

            # Each participant responds
            for participant in self.participants:
                self.session.start_turn(participant.name)
                response = participant.generate_response(
                    topic=self.config.discussion.topic,
                    conversation_history=self.session.all_messages,
                )
                self._add_message(
                    agent_name=response.agent_name,
                    content=response.content,
                    message_type=MessageType.CONTRIBUTION,
                    metadata=response.metadata,
                )
                self.session.complete_turn()

            # Round summary by moderator
            summary_response = self.moderator.generate_round_summary(
                topic=self.config.discussion.topic,
                conversation_history=self.session.all_messages,
                round_number=current_round,
            )
            self.session.complete_round(summary_response.content)

            # Check if discussion should continue
            should_continue = self.moderator.should_continue_discussion(
                topic=self.config.discussion.topic,
                conversation_history=self.session.all_messages,
                current_round=current_round,
                max_rounds=max_rounds,
            )

            if not should_continue:
                break

            current_round += 1

        self.session.transition_state(DiscussionState.CONCLUDING)

    def _run_closing_phase(self):
        """Run the closing phase of the discussion."""
        self.session.start_round()

        # Moderator closes the discussion
        self.session.start_turn(self.moderator.name)
        closing = self.moderator.generate_closing(
            topic=self.config.discussion.topic,
            conversation_history=self.session.all_messages,
        )
        self._add_message(
            agent_name=closing.agent_name,
            content=closing.content,
            message_type=MessageType.CLOSING,
            metadata=closing.metadata,
        )
        self.session.complete_turn()

        self.session.complete_round("クロージングラウンド完了")
        self.session.transition_state(DiscussionState.COMPLETED)

    def _run_evaluation_phase(self) -> Optional[EvaluationResult]:
        """Run the evaluation phase."""
        if not self.evaluator:
            return None

        evaluation = self.evaluator.evaluate_discussion(
            topic=self.config.discussion.topic,
            conversation_history=self.session.all_messages,
        )

        # Add evaluation to session metadata
        self.session.metadata["evaluation"] = evaluation.to_dict()

        return evaluation


def run_discussion(
    topic: str,
    provider: str = "openai",
    model: Optional[str] = None,
    participant_personas: Optional[List[dict]] = None,
    max_rounds: int = 5,
    output_language: str = "ja",
    enable_search: bool = False,
    search_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    message_callback: Optional[Callable[[Message], None]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a discussion with sensible defaults.

    Args:
        topic: The topic to discuss
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (uses default if not specified)
        participant_personas: List of dicts with 'name' and 'persona' keys
        max_rounds: Maximum number of discussion rounds
        output_language: Output language code
        enable_search: If True, participants will search the web for information
        search_config: Optional search configuration for research participants
        progress_callback: Optional callback for progress updates
        message_callback: Optional callback for new messages

    Returns:
        Dictionary with discussion results
    """
    config = create_config(
        topic=topic,
        provider=provider,
        model=model,
        participant_personas=participant_personas,
        max_rounds=max_rounds,
        output_language=output_language,
        enable_search=enable_search,
        search_config=search_config,
    )

    discussion = MultiAgentDiscussion(config)
    return discussion.run(
        progress_callback=progress_callback,
        message_callback=message_callback,
    )
