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


def _reclose_tail(text: str):
    """(text, old tail, new tail) with the reply's final run of closing brackets replaced by
    the one run that closes what is still open; None when that run is already right or the
    text before it is not well nested.

    qwen3-30b-a3b miscounts the closers that end its JSON after a nested argument (dev6
    run 3: '"laps": 3}]}}}]}', one brace too many after compare_plans' candidates; now and
    then one too few, or a stray one after the end). Only that final run changes: everything
    before it is kept as written and must be well nested, and nothing but closers follows,
    so exactly one run of closers completes it and no key or value can change level.
    """

    pairs = {"}": "{", "]": "["}
    end = len(text)
    while end and text[end - 1] in "}] \t\r\n":
        end -= 1
    stack, in_string, escaped = [], False, False

    for ch in text[:end]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return None

    old = "".join(text[end:].split())
    new = "".join("}" if c == "{" else "]" for c in reversed(stack))
    if in_string or old == new:
        return None
    return text[:end] + new, old, new


def _parse_reclosed(text: str, exc: json.JSONDecodeError):
    """(payload, note) of the reply's JSON after _reclose_tail; raises `exc`, the original
    error, when that does not apply or the result still does not parse."""

    body = re.sub(r"\s*```$", "", text[text.find("{"):]) if "{" in text else ""
    fixed = _reclose_tail(body)

    if fixed is None:
        raise exc

    try:
        payload = json.loads(fixed[0])
    except json.JSONDecodeError:
        raise exc from None

    return payload, f"final closing brackets {fixed[1]!r} read as {fixed[2]!r}"


class VoyagerBackend(ModelBackend):
    """Model backend for ASU Research Computing Voyager."""

    name = "voyager"

    def __init__(self):
        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        self.base_url = os.environ.get(
            "VOYAGER_BASE_URL",
            "https://openai.rc.asu.edu/v1",
        )

        self.api_key = os.environ.get(
            "VOYAGER_API_KEY",
            "EMPTY",
        )

        self.model_id = os.environ.get(
            "VOYAGER_MODEL",
            "qwen3-30b-a3b",
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
            temperature=float(
                os.environ.get(
                    "VOYAGER_TEMPERATURE",
                    # Some OpenAI-compatible servers treat 0 as "use the
                    # server default"; a tiny positive value asks for greedy
                    # decoding explicitly.
                    "1e-9",
                )
            ),
        )

        text = response.choices[0].message.content or ""
        text = text.strip()

        # ------------------------------------------------------------------
        # Parse requested tool calls
        # ------------------------------------------------------------------

        payload = None
        parse_error = repair = None

        try:
            try:
                payload = _extract_json(text)
            except json.JSONDecodeError as exc:
                # a reply cut off at the token cap is unfinished, not mis-closed: never repaired
                if response.choices[0].finish_reason == "length":
                    raise
                payload, repair = _parse_reclosed(text, exc)

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
            parse_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, json.JSONDecodeError):
                # where it failed, so the model can see what to correct
                parse_error += (f"; the reply up to there ends {exc.doc[max(exc.pos - 60, 0):exc.pos]!r}, "
                                f"followed by {exc.doc[exc.pos:exc.pos + 20]!r}")

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
            raw=response,
            parse_error=parse_error,
            parse_repair=repair,
        )