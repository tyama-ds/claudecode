"""Message data classes for conversation tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class MessageType(str, Enum):
    """Types of messages in a discussion."""
    OPENING = "opening"
    CONTRIBUTION = "contribution"
    REBUTTAL = "rebuttal"
    AGREEMENT = "agreement"
    MODERATION = "moderation"
    EVALUATION = "evaluation"
    CLOSING = "closing"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a single message in the discussion."""
    agent_name: str
    content: str
    message_type: MessageType = MessageType.CONTRIBUTION
    timestamp: datetime = field(default_factory=datetime.now)
    round_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "round_number": self.round_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            content=data["content"],
            message_type=MessageType(data.get("message_type", "contribution")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            round_number=data.get("round_number", 0),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation of the message."""
        return f"[{self.agent_name}]: {self.content}"


@dataclass
class Turn:
    """Represents a turn in the discussion (one agent's complete contribution)."""
    turn_number: int
    agent_name: str
    messages: list = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_message(self, message: Message):
        """Add a message to this turn."""
        self.messages.append(message)

    def complete(self):
        """Mark this turn as complete."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of this turn in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary."""
        return {
            "turn_number": self.turn_number,
            "agent_name": self.agent_name,
            "messages": [m.to_dict() for m in self.messages],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Create turn from dictionary."""
        turn = cls(
            turn_number=data["turn_number"],
            agent_name=data["agent_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        )
        turn.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return turn


@dataclass
class Round:
    """Represents a round in the discussion (all agents have spoken)."""
    round_number: int
    turns: list = field(default_factory=list)
    summary: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_turn(self, turn: Turn):
        """Add a turn to this round."""
        self.turns.append(turn)

    def complete(self, summary: str = ""):
        """Mark this round as complete."""
        self.end_time = datetime.now()
        self.summary = summary

    @property
    def all_messages(self) -> list:
        """Get all messages in this round."""
        messages = []
        for turn in self.turns:
            messages.extend(turn.messages)
        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert round to dictionary."""
        return {
            "round_number": self.round_number,
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Round":
        """Create round from dictionary."""
        round_obj = cls(
            round_number=data["round_number"],
            summary=data.get("summary", ""),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        )
        round_obj.turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return round_obj
