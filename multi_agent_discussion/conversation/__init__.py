"""Conversation management module for multi-agent discussion."""

from .message import Message, MessageType, Turn, Round
from .session import DiscussionSession

__all__ = [
    "Message",
    "MessageType",
    "Turn",
    "Round",
    "DiscussionSession",
]
