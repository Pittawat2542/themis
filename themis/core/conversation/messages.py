"""Single message in a conversation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """Single message in a conversation.

    Attributes:
        role: Message role (system/user/assistant/tool)
        content: Message text content
        metadata: Additional metadata (tool calls, timestamps, etc.)
        timestamp: Unix timestamp when message was created
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
