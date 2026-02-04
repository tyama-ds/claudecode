"""Tests for session management module."""

import json
import tempfile
from pathlib import Path

import pytest

from multi_agent_discussion.conversation import (
    Message,
    MessageType,
    Turn,
    Round,
    DiscussionSession,
)
from multi_agent_discussion.config import DiscussionState


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(
            agent_name="TestAgent",
            content="Test content",
            message_type=MessageType.CONTRIBUTION,
        )
        assert msg.agent_name == "TestAgent"
        assert msg.content == "Test content"
        assert msg.message_type == MessageType.CONTRIBUTION

    def test_message_serialization(self):
        """Test message can be serialized to dict."""
        msg = Message(
            agent_name="TestAgent",
            content="Test content",
            round_number=1,
        )
        data = msg.to_dict()
        assert data["agent_name"] == "TestAgent"
        assert data["content"] == "Test content"
        assert data["round_number"] == 1

    def test_message_deserialization(self):
        """Test message can be deserialized from dict."""
        original = Message(
            agent_name="TestAgent",
            content="Test content",
            message_type=MessageType.MODERATION,
        )
        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.agent_name == original.agent_name
        assert restored.content == original.content
        assert restored.message_type == original.message_type

    def test_message_str(self):
        """Test string representation."""
        msg = Message(agent_name="Agent", content="Hello")
        assert "[Agent]: Hello" == str(msg)


class TestTurn:
    """Tests for Turn class."""

    def test_create_turn(self):
        """Test creating a turn."""
        turn = Turn(turn_number=1, agent_name="Agent")
        assert turn.turn_number == 1
        assert turn.agent_name == "Agent"
        assert len(turn.messages) == 0

    def test_add_message(self):
        """Test adding messages to turn."""
        turn = Turn(turn_number=1, agent_name="Agent")
        msg = Message(agent_name="Agent", content="Hello")
        turn.add_message(msg)
        assert len(turn.messages) == 1

    def test_complete_turn(self):
        """Test completing a turn."""
        turn = Turn(turn_number=1, agent_name="Agent")
        assert turn.end_time is None
        turn.complete()
        assert turn.end_time is not None

    def test_turn_serialization(self):
        """Test turn serialization."""
        turn = Turn(turn_number=1, agent_name="Agent")
        turn.add_message(Message(agent_name="Agent", content="Hello"))
        turn.complete()

        data = turn.to_dict()
        restored = Turn.from_dict(data)

        assert restored.turn_number == turn.turn_number
        assert len(restored.messages) == 1


class TestRound:
    """Tests for Round class."""

    def test_create_round(self):
        """Test creating a round."""
        round_obj = Round(round_number=1)
        assert round_obj.round_number == 1
        assert len(round_obj.turns) == 0

    def test_add_turn(self):
        """Test adding turns to round."""
        round_obj = Round(round_number=1)
        turn = Turn(turn_number=1, agent_name="Agent")
        round_obj.add_turn(turn)
        assert len(round_obj.turns) == 1

    def test_all_messages(self):
        """Test getting all messages from round."""
        round_obj = Round(round_number=1)

        turn1 = Turn(turn_number=1, agent_name="Agent1")
        turn1.add_message(Message(agent_name="Agent1", content="Hello"))
        turn1.add_message(Message(agent_name="Agent1", content="World"))

        turn2 = Turn(turn_number=2, agent_name="Agent2")
        turn2.add_message(Message(agent_name="Agent2", content="Hi"))

        round_obj.add_turn(turn1)
        round_obj.add_turn(turn2)

        messages = round_obj.all_messages
        assert len(messages) == 3


class TestDiscussionSession:
    """Tests for DiscussionSession class."""

    def test_create_session(self):
        """Test creating a session."""
        session = DiscussionSession(topic="Test topic")
        assert session.topic == "Test topic"
        assert session.state == DiscussionState.INITIALIZED
        assert len(session.session_id) == 8

    def test_start_round(self):
        """Test starting a new round."""
        session = DiscussionSession(topic="Test")
        round_obj = session.start_round()
        assert round_obj.round_number == 1
        assert session.current_round_number == 1

    def test_add_message(self):
        """Test adding messages to session."""
        session = DiscussionSession(topic="Test")
        session.start_round()
        session.start_turn("Agent")

        msg = session.add_message(
            agent_name="Agent",
            content="Hello",
            message_type=MessageType.CONTRIBUTION,
        )

        assert msg.agent_name == "Agent"
        assert session.message_count == 1

    def test_complete_round(self):
        """Test completing a round."""
        session = DiscussionSession(topic="Test")
        session.start_round()
        session.add_message("Agent", "Hello")
        session.complete_round("Round summary")

        assert len(session.rounds) == 1
        assert session.rounds[0].summary == "Round summary"

    def test_state_transitions(self):
        """Test state transitions."""
        session = DiscussionSession(topic="Test")
        assert session.state == DiscussionState.INITIALIZED

        session.transition_state(DiscussionState.OPENING)
        assert session.state == DiscussionState.OPENING

        session.transition_state(DiscussionState.DISCUSSING)
        assert session.state == DiscussionState.DISCUSSING

    def test_invalid_state_transition(self):
        """Test invalid state transition raises error."""
        session = DiscussionSession(topic="Test")
        with pytest.raises(ValueError):
            session.transition_state(DiscussionState.COMPLETED)

    def test_session_serialization(self):
        """Test session serialization to dict."""
        session = DiscussionSession(topic="Test")
        session.start_round()
        session.add_message("Agent", "Hello")
        session.complete_round()

        data = session.to_dict()
        assert data["topic"] == "Test"
        assert len(data["rounds"]) == 1

    def test_session_deserialization(self):
        """Test session deserialization from dict."""
        session = DiscussionSession(topic="Test", participants=["A", "B"])
        session.start_round()
        session.add_message("A", "Hello")
        session.complete_round()

        data = session.to_dict()
        restored = DiscussionSession.from_dict(data)

        assert restored.topic == session.topic
        assert restored.session_id == session.session_id
        assert len(restored.rounds) == 1

    def test_save_and_load(self):
        """Test saving and loading session."""
        session = DiscussionSession(topic="Test")
        session.start_round()
        session.add_message("Agent", "Hello")
        session.complete_round()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_session.json"
            session.save(filepath)

            loaded = DiscussionSession.load(filepath)
            assert loaded.topic == session.topic
            assert loaded.session_id == session.session_id

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        session = DiscussionSession(topic="Test")
        session.start_round()

        for i in range(15):
            session.add_message("Agent", f"Message {i}")

        recent = session.get_recent_messages(count=5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"

    def test_get_messages_by_agent(self):
        """Test filtering messages by agent."""
        session = DiscussionSession(topic="Test")
        session.start_round()
        session.add_message("Agent1", "Hello")
        session.add_message("Agent2", "Hi")
        session.add_message("Agent1", "How are you?")

        agent1_msgs = session.get_messages_by_agent("Agent1")
        assert len(agent1_msgs) == 2

    def test_generate_transcript(self):
        """Test transcript generation."""
        session = DiscussionSession(topic="Test topic")
        session.participants = ["Agent1", "Agent2"]
        session.start_round()
        session.add_message("Agent1", "Hello", MessageType.OPENING)
        session.complete_round()

        transcript = session.generate_transcript()
        assert "Test topic" in transcript
        assert "Agent1" in transcript
        assert "Hello" in transcript
