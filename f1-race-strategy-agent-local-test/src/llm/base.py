"""Provider-neutral types for the strategy agent's model backends."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class ModelBackend:
    """Small interface implemented by every model provider."""

    name = "base"

    def run_turn(self, *, system_prompt, messages, tools, max_tokens):
        raise NotImplementedError
