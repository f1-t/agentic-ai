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
    parse_error: str | None = None   # why a text backend found no tool call in its reply, if it failed to parse
    parse_repair: str | None = None  # what a text backend corrected in the reply's JSON before parsing it


class ModelBackend:
    """Small interface implemented by every model provider."""

    name = "base"

    def run_turn(self, *, system_prompt, messages, tools, max_tokens):
        raise NotImplementedError
