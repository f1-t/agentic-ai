"""ASU RC Voyager backend.

Uses ASU Research Computing's OpenAI-compatible Voyager endpoint while keeping
tool calls provider-independent.

The model does NOT use a provider-specific tool API. Instead, it emits JSON
describing requested tool calls, which is parsed by the agent loop.
"""

import json
import os
import re
import uuid

from openai import OpenAI

from .base import ModelBackend, ModelTurn, ToolCall


_JSON_INSTRUCTION = """
You do not have a native tool API.

On each turn, respond with JSON ONLY using one of these forms:

Single tool call:
{"tool_calls":[{"name":"TOOL_NAME","arguments":{...}}]}

Multiple independent tool calls are allowed:
{"tool_calls":[{"name":"TOOL_1","arguments":{...}},{"name":"TOOL_2","arguments":{...}}]}

Do not wrap the JSON in markdown fences.
Do not include prose outside the JSON.
Do not invent tool names.
Use the tool results already present in the conversation before deciding the
next action.

When finished, call submit_recommendation as a tool.
Never return a prose-only final answer.
"""


def _extract_json(text: str):
    """Extract a JSON object from model output.

    Voyager-hosted instruct models may occasionally wrap JSON in markdown or
    add a small amount of text around it, so retain the same tolerant parsing
    behavior as the local Hugging Face backend.
    """

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        # Fallback for models that add text around otherwise valid JSON.
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise


class VoyagerBackend(ModelBackend):
    """Model backend for ASU Research Computing Voyager."""

    name = "voyager"

    def __init__(self):
        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        self.base_url = os.environ.get(
            "VOYAGER_BASE_URL",
            "https://voyager.rc.asu.edu/v1",
        )

        self.api_key = os.environ.get(
            "VOYAGER_API_KEY",
            "EMPTY",
        )

        self.model_id = os.environ.get(
            "VOYAGER_MODEL",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        )

        # ------------------------------------------------------------------
        # OpenAI-compatible client
        # ------------------------------------------------------------------

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _tool_text(self, tools):
        """Serialize agent tools into the prompt."""

        compact = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

        return json.dumps(
            compact,
            ensure_ascii=False,
        )

    def run_turn(
        self,
        *,
        system_prompt,
        messages,
        tools,
        max_tokens,
    ):
        """Run one model turn through Voyager."""

        # ------------------------------------------------------------------
        # Construct system prompt
        # ------------------------------------------------------------------

        system_content = (
            system_prompt
            + "\n"
            + _JSON_INSTRUCTION
            + "\nAvailable tools:\n"
            + self._tool_text(tools)
        )

        chat = [
            {
                "role": "system",
                "content": system_content,
            }
        ]

        chat.extend(messages)

        # ------------------------------------------------------------------
        # Model request
        # ------------------------------------------------------------------

        local_max = int(
            os.environ.get(
                "VOYAGER_MAX_TOKENS",
                min(max_tokens, 1024),
            )
        )

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=chat,
            max_tokens=local_max,
            temperature=0.0,
        )

        text = response.choices[0].message.content or ""
        text = text.strip()

        # ------------------------------------------------------------------
        # Parse requested tool calls
        # ------------------------------------------------------------------

        payload = None

        try:
            payload = _extract_json(text)

            raw_calls = payload.get(
                "tool_calls",
                [],
            )

            calls = [
                ToolCall(
                    id=f"voyager_{uuid.uuid4().hex[:10]}",
                    name=c["name"],
                    arguments=c.get(
                        "arguments",
                        {},
                    ),
                )
                for c in raw_calls
            ]

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:

            calls = []

            text = (
                f"{text}\n"
                f"[tool-call parse error: {exc}]"
            )

        # ------------------------------------------------------------------
        # Return provider-independent model turn
        # ------------------------------------------------------------------

        return ModelTurn(
            text=text,
            tool_calls=calls,
            raw=payload,
        )