"""Anthropic implementation of the provider-neutral model interface."""
import os
from anthropic import Anthropic

from .base import ModelBackend, ModelTurn, ToolCall


class AnthropicBackend(ModelBackend):
    name = "anthropic"

    def __init__(self):
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        self.client = Anthropic()

    def run_turn(self, *, system_prompt, messages, tools, max_tokens):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        text_parts = []
        calls = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return ModelTurn(text="\n".join(text_parts), tool_calls=calls, raw=resp)
