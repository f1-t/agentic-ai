"""A minimal, explicit, provider-neutral F1 race-strategy agent loop.

The model is selected with LLM_BACKEND. The F1 tools and plan -> act -> observe ->
refine loop stay identical across Anthropic and local Hugging Face models.
"""
import json
import os

from dotenv import load_dotenv

from . import tools
from .llm import create_backend

load_dotenv()

MAX_STEPS = 10

SYSTEM_PROMPT = """You are a Formula 1 race strategist. Your job is to recommend a \
pit-stop / tyre strategy for one driver in one race, and justify it with numbers.

You have tools to read race info, a driver's actual stints, per-compound tyre \
degradation, and a strategy simulator that estimates total race time for any \
proposed stint plan. The simulator is the source of truth for comparing strategies.

Work like a strategist:
1. Get the race info and the driver's starting compound.
2. Estimate degradation for each available compound.
3. Propose a focused set of about 4-8 well-chosen candidate strategies (both \
one-stop and two-stop) that START on the driver's actual starting compound, varying \
the first-stint length and the compound split. Simulate each. Do NOT do an \
exhaustive search - a handful of sensible candidates is enough.
4. Compare the simulated total times and pick the fastest valid strategy.
5. As soon as you have a clear best candidate, call submit_recommendation. Do not \
keep simulating once you have one.

Rules: every strategy's stint laps must sum to the race length and use at least two \
different compounds. If the simulator returns an error, fix the plan and try again. \
Base your recommendation only on simulated total times, not intuition."""

SUBMIT_TOOL = {
    "name": "submit_recommendation",
    "description": "Submit your FINAL strategy recommendation. Call this once, after you have simulated and compared candidates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "num_stops": {"type": "integer", "description": "Recommended number of pit stops (1 or 2)."},
            "first_pit_lap": {"type": "integer", "description": "Lap of the first pit stop (= laps in the opening stint)."},
            "stint_plan": {
                "type": "array",
                "description": "Full plan, e.g. [[\"SOFT\", 28], [\"SOFT\", 28], [\"MEDIUM\", 10]]",
                "items": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "minItems": 2, "maxItems": 2},
            },
            "total_time_s": {"type": "number", "description": "Simulated total race time of the chosen strategy."},
            "rationale": {"type": "string", "description": "One or two sentences on why this strategy wins."},
        },
        "required": ["num_stops", "first_pit_lap", "stint_plan", "rationale"],
        "additionalProperties": False,
    },
}


def _api_tools():
    specs = [{k: t[k] for k in ("name", "description", "input_schema")} for t in tools.TOOL_SPECS]
    return specs + [SUBMIT_TOOL]


def _dispatch(name, args):
    """Dispatch a model tool call using only arguments declared by its schema.

    Local text-generation models can occasionally emit plausible but undeclared
    keys (for example ``race``). Provider-native tool APIs often reject those
    before execution; for the provider-neutral path we enforce the same boundary
    here so an extra key cannot crash a tool callable.
    """
    for t in tools.TOOL_SPECS:
        if t["name"] != name:
            continue

        schema = t.get("input_schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        clean_args = {k: v for k, v in args.items() if k in properties}
        missing = [k for k in required if k not in clean_args]
        if missing:
            return {
                "error": f"Missing required argument(s) for {name}: {', '.join(missing)}"
            }

        try:
            return t["fn"](**clean_args)
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}

    return {"error": f"Unknown tool '{name}'"}


def run_agent(question, verbose=True):
    """Drive the provider-neutral loop. Returns (recommendation, trajectory)."""
    backend = create_backend()
    api_tools = _api_tools()
    messages = [{"role": "user", "content": question}]
    trajectory = []

    if verbose:
        model_name = getattr(backend, "model", getattr(backend, "model_id", "unknown"))
        device = getattr(backend, "device", None)
        suffix = f" on {device}" if device else ""
        print(f"  [backend] {backend.name}: {model_name}{suffix}")

    for step in range(MAX_STEPS):
        turn = backend.run_turn(
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            tools=api_tools,
            max_tokens=8000,
        )

        if turn.text.strip() and verbose:
            print(f"  [think] {turn.text.strip()[:300]}")

        if not turn.tool_calls:
            return None, trajectory

        # Store a provider-neutral assistant turn. AnthropicBackend converts the
        # history it needs internally; local HF consumes this textual transcript.
        if backend.name == "anthropic":
            messages.append({"role": "assistant", "content": turn.raw.content})
        else:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                    ]
                }),
            })

        results = []
        rec = None
        for call in turn.tool_calls:
            if call.name == "submit_recommendation":
                rec = dict(call.arguments)
                if backend.name == "anthropic":
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": "Recommendation recorded."})
            else:
                out = _dispatch(call.name, dict(call.arguments))
                trajectory.append({"step": step + 1, "tool": call.name, "input": dict(call.arguments), "output": out})
                if verbose:
                    print(f"  [tool] {call.name}({json.dumps(call.arguments)}) -> {json.dumps(out)[:200]}")
                if backend.name == "anthropic":
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(out)})
                else:
                    results.append({"tool": call.name, "result": out})

        if rec is not None:
            if verbose:
                print(f"  [submit] {json.dumps(rec)}")
            return rec, trajectory

        if backend.name == "anthropic":
            messages.append({"role": "user", "content": results})
        else:
            messages.append({"role": "user", "content": "Tool results:\n" + json.dumps(results, ensure_ascii=False)})

    return None, trajectory
