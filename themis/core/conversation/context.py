"""Conversation context and history."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from themis.core.conversation.messages import Message, MessageRole
from themis.generation import templates


@dataclass
class ConversationContext:
    """Maintains conversation state across turns.

    This class manages the conversation history and provides utilities
    for rendering conversations as prompts.
    """

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))

    def get_history(self, max_turns: int | None = None) -> list[Message]:
        """Get conversation history."""
        if max_turns is None:
            return list(self.messages)
        return self.messages[-max_turns:]

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Get all messages with a specific role."""
        return [msg for msg in self.messages if msg.role == role]

    def to_prompt(self, template: templates.PromptTemplate | None = None) -> str:
        """Render conversation to prompt string."""
        if template is not None:
            return template.render(messages=self.messages)

        # Default format: role-prefixed messages
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        """Create conversation from dictionary."""
        context = cls(metadata=data.get("metadata", {}))
        for msg_data in data.get("messages", []):
            context.messages.append(
                Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    metadata=msg_data.get("metadata", {}),
                    timestamp=msg_data.get("timestamp", time.time()),
                )
            )
        return context

    def __len__(self) -> int:
        """Return number of messages in conversation."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Return a human-readable summary of the conversation."""
        role_counts = {}
        for msg in self.messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
        parts = ", ".join(f"{r}={c}" for r, c in sorted(role_counts.items()))
        return f"ConversationContext(messages={len(self.messages)}, {parts})"

    def clear(self) -> None:
        """Remove all messages while preserving metadata."""
        self.messages.clear()

    def estimate_tokens(self, chars_per_token: float = 4.0) -> int:
        """Estimate total token count using a character-count heuristic."""
        total_chars = sum(len(msg.content) for msg in self.messages)
        return int(total_chars / chars_per_token)

    def truncate(self, max_messages: int, *, keep_system: bool = True) -> None:
        """Truncate conversation to the most recent messages."""
        if len(self.messages) <= max_messages:
            return

        if keep_system:
            system_prefix = []
            rest = []
            for msg in self.messages:
                if not rest and msg.role == "system":
                    system_prefix.append(msg)
                else:
                    rest.append(msg)
            # Keep system prefix + last N non-system messages
            keep_count = max(0, max_messages - len(system_prefix))
            self.messages = (
                system_prefix + rest[-keep_count:] if keep_count else system_prefix
            )
        else:
            self.messages = self.messages[-max_messages:]
