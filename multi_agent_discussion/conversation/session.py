"""Session management for multi-agent discussions."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .message import Message, MessageType, Turn, Round
from ..config import DiscussionState


@dataclass
class DiscussionSession:
    """Manages the state and history of a discussion session."""
    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    topic: str = ""
    state: DiscussionState = DiscussionState.INITIALIZED
    rounds: List[Round] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal tracking
    _current_round: Optional[Round] = field(default=None, repr=False)
    _current_turn: Optional[Turn] = field(default=None, repr=False)
    _turn_counter: int = field(default=0, repr=False)

    @property
    def current_round_number(self) -> int:
        """Get the current round number."""
        return len(self.rounds) + (1 if self._current_round else 0)

    @property
    def all_messages(self) -> List[Message]:
        """Get all messages from all rounds."""
        messages = []
        for round_obj in self.rounds:
            messages.extend(round_obj.all_messages)
        if self._current_round:
            messages.extend(self._current_round.all_messages)
        # Include messages from current turn (not yet added to round)
        if self._current_turn:
            messages.extend(self._current_turn.messages)
        return messages

    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.all_messages)

    def start_round(self) -> Round:
        """Start a new round."""
        if self._current_round:
            self._complete_current_round()

        # Calculate round number before creating (1-indexed)
        round_number = len(self.rounds) + 1
        self._current_round = Round(round_number=round_number)
        self.updated_at = datetime.now()
        return self._current_round

    def _complete_current_round(self, summary: str = ""):
        """Complete the current round and add to history."""
        if self._current_round:
            self._current_round.complete(summary)
            self.rounds.append(self._current_round)
            self._current_round = None

    def start_turn(self, agent_name: str) -> Turn:
        """Start a new turn for an agent."""
        if self._current_turn:
            self._current_turn.complete()
            if self._current_round:
                self._current_round.add_turn(self._current_turn)

        self._turn_counter += 1
        self._current_turn = Turn(
            turn_number=self._turn_counter,
            agent_name=agent_name,
        )
        self.updated_at = datetime.now()
        return self._current_turn

    def add_message(
        self,
        agent_name: str,
        content: str,
        message_type: MessageType = MessageType.CONTRIBUTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the current turn."""
        message = Message(
            agent_name=agent_name,
            content=content,
            message_type=message_type,
            round_number=self.current_round_number,
            metadata=metadata or {},
        )

        if self._current_turn:
            self._current_turn.add_message(message)
        elif self._current_round:
            # Create an ad-hoc turn
            turn = Turn(turn_number=self._turn_counter + 1, agent_name=agent_name)
            turn.add_message(message)
            turn.complete()
            self._current_round.add_turn(turn)
            self._turn_counter += 1

        self.updated_at = datetime.now()
        return message

    def complete_turn(self):
        """Complete the current turn."""
        if self._current_turn:
            self._current_turn.complete()
            if self._current_round:
                self._current_round.add_turn(self._current_turn)
            self._current_turn = None
        self.updated_at = datetime.now()

    def complete_round(self, summary: str = ""):
        """Complete the current round."""
        self.complete_turn()  # Complete any pending turn
        self._complete_current_round(summary)
        self.updated_at = datetime.now()

    def transition_state(self, new_state: DiscussionState):
        """Transition to a new discussion state."""
        valid_transitions = {
            DiscussionState.INITIALIZED: [DiscussionState.OPENING],
            DiscussionState.OPENING: [DiscussionState.DISCUSSING],
            DiscussionState.DISCUSSING: [DiscussionState.DISCUSSING, DiscussionState.CONCLUDING],
            DiscussionState.CONCLUDING: [DiscussionState.COMPLETED],
            DiscussionState.COMPLETED: [],
        }

        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(
                f"Invalid state transition: {self.state} -> {new_state}"
            )

        self.state = new_state
        self.updated_at = datetime.now()

    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages."""
        all_msgs = self.all_messages
        return all_msgs[-count:] if len(all_msgs) > count else all_msgs

    def get_messages_by_agent(self, agent_name: str) -> List[Message]:
        """Get all messages from a specific agent."""
        return [m for m in self.all_messages if m.agent_name == agent_name]

    def get_round_messages(self, round_number: int) -> List[Message]:
        """Get all messages from a specific round."""
        if round_number <= len(self.rounds):
            return self.rounds[round_number - 1].all_messages
        elif self._current_round and round_number == self.current_round_number:
            return self._current_round.all_messages
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        # Complete any pending round before serialization
        data = {
            "session_id": self.session_id,
            "topic": self.topic,
            "state": self.state.value,
            "rounds": [r.to_dict() for r in self.rounds],
            "participants": self.participants,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

        # Include current round if exists
        if self._current_round:
            data["current_round"] = self._current_round.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscussionSession":
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            topic=data["topic"],
            state=DiscussionState(data["state"]),
            participants=data.get("participants", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
        session.rounds = [Round.from_dict(r) for r in data.get("rounds", [])]

        # Restore current round if exists
        if "current_round" in data:
            session._current_round = Round.from_dict(data["current_round"])

        # Calculate turn counter
        session._turn_counter = sum(
            len(r.turns) for r in session.rounds
        )
        if session._current_round:
            session._turn_counter += len(session._current_round.turns)

        return session

    def save(self, filepath: Optional[Path] = None, session_dir: str = "./discussion_sessions"):
        """Save session to JSON file."""
        if filepath is None:
            dir_path = Path(session_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            filepath = dir_path / f"session_{self.session_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "DiscussionSession":
        """Load session from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def generate_transcript(self) -> str:
        """Generate a human-readable transcript of the discussion."""
        lines = [
            f"# 議論トランスクリプト",
            f"",
            f"**トピック**: {self.topic}",
            f"**セッションID**: {self.session_id}",
            f"**参加者**: {', '.join(self.participants)}",
            f"**開始時刻**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "---",
            "",
        ]

        for round_obj in self.rounds:
            lines.append(f"## ラウンド {round_obj.round_number}")
            lines.append("")

            for message in round_obj.all_messages:
                lines.append(f"**[{message.agent_name}]** ({message.message_type.value})")
                lines.append(f"{message.content}")
                lines.append("")

            if round_obj.summary:
                lines.append(f"*ラウンドまとめ: {round_obj.summary}*")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Include current round if exists
        if self._current_round:
            lines.append(f"## ラウンド {self._current_round.round_number} (進行中)")
            lines.append("")
            for message in self._current_round.all_messages:
                lines.append(f"**[{message.agent_name}]** ({message.message_type.value})")
                lines.append(f"{message.content}")
                lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"DiscussionSession(id={self.session_id}, topic='{self.topic[:30]}...', "
            f"state={self.state.value}, rounds={len(self.rounds)}, "
            f"messages={self.message_count})"
        )
