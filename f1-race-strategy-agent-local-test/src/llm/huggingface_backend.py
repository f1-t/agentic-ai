"""Local Hugging Face/Transformers backend, including Apple Silicon MPS support.

Tool calls are requested as JSON and parsed by our code instead of relying on a
provider-specific tool API. This keeps the F1 agent loop model-independent.
"""
import json
import os
import re
import uuid

from .base import ModelBackend, ModelTurn, ToolCall


_JSON_INSTRUCTION = """
You do not have a native tool API. On each turn, respond with JSON ONLY using one of these forms:

Single tool call:
{"tool_calls":[{"name":"TOOL_NAME","arguments":{...}}]}

Multiple independent tool calls are allowed:
{"tool_calls":[{"name":"TOOL_1","arguments":{...}},{"name":"TOOL_2","arguments":{...}}]}

Do not wrap the JSON in markdown fences. Do not invent tool names. Use the tool results in the conversation before deciding the next action. When finished, call submit_recommendation as a tool; never return a prose-only final answer.
"""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Small instruct models sometimes add a sentence around otherwise valid JSON.
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class HuggingFaceBackend(ModelBackend):
    name = "huggingface"

    def __init__(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend needs torch and transformers. "
                "Install them with: pip install -r requirements-hf.txt"
            ) from exc

        self.torch = torch
        self.model_id = os.environ.get("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        requested = os.environ.get("HF_DEVICE", "auto").lower()
        if requested == "auto":
            if torch.backends.mps.is_available():
                requested = "mps"
            elif torch.cuda.is_available():
                requested = "cuda"
            else:
                requested = "cpu"
        self.device = requested

        token = os.environ.get("HF_TOKEN") or None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        # float16 is broadly supported on Apple Silicon; CPU stays float32.
        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _tool_text(self, tools):
        compact = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]
        return json.dumps(compact, ensure_ascii=False)

    def run_turn(self, *, system_prompt, messages, tools, max_tokens):
        chat = [{
            "role": "system",
            "content": system_prompt + _JSON_INSTRUCTION + "\nAvailable tools:\n" + self._tool_text(tools),
        }]
        chat.extend(messages)

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        local_max = int(os.environ.get("HF_MAX_NEW_TOKENS", min(max_tokens, 1024)))
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=local_max,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            payload = _extract_json(text)
            raw_calls = payload.get("tool_calls", [])
            calls = [
                ToolCall(
                    id=f"local_{uuid.uuid4().hex[:10]}",
                    name=c["name"],
                    arguments=c.get("arguments", {}),
                )
                for c in raw_calls
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            calls = []
            text = f"{text}\n[tool-call parse error: {exc}]"

        return ModelTurn(text=text, tool_calls=calls, raw=payload if 'payload' in locals() else None)
