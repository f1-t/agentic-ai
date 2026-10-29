"""Provider-neutral, snapshot-based F1 strategy agent loop."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import tools
from .llm import create_backend
from .snapshot import RaceSnapshot

load_dotenv()

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_TOOL_CALLS = 24

SYSTEM_PROMPT = """You are a Formula 1 race strategist making a MID-RACE decision from a snapshot in time.

The snapshot is the only current race state. Do not assume a fixed race, driver, lap, or starting tyre. Use tools when they reduce decision uncertainty.

IMPORTANT: tool outputs are estimates. Ranges, probabilities, confidence labels and assumptions represent uncertainty; the simulator is NOT ground truth. Compare plausible outcomes and downside risk rather than blindly choosing the smallest median number.

Your task is to choose the immediate action: PIT now (and which compound) or STAY OUT for a stated number of laps. Consider position, gaps/traffic, tyre age/degradation, pit-loss range, weather/track status and nearby competitors. Prefer a small number of informative tool calls.

If a tool reports an invalid plan or arguments, use its error/hint to correct the call. Do not repeat the same invalid call. You have a finite step/tool budget, so submit once the evidence is sufficient.

Finish by calling submit_recommendation. Never finish with prose alone."""

SUBMIT_TOOL = {
    "name": "submit_recommendation",
    "description": "Submit the final immediate mid-race strategy decision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "description": "PIT or STAY_OUT"},
            "compound": {"type": ["string", "null"], "description": "Compound if pitting; otherwise null."},
            "reassess_after_laps": {"type": "integer", "description": "When to reassess if staying out, or after the pit."},
            "confidence": {"type": "number", "description": "0-1 confidence in this decision."},
            "rationale": {"type": "string", "description": "Short evidence-based rationale acknowledging material uncertainty."},
            "key_risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "reassess_after_laps", "confidence", "rationale", "key_risks"],
        "additionalProperties": False,
    },
}


def _api_tools():
    specs = [{k: t[k] for k in ("name", "description", "input_schema")} for t in tools.TOOL_SPECS]
    return specs + [SUBMIT_TOOL]


def _dispatch(name, args):
    for spec in tools.TOOL_SPECS:
        if spec["name"] != name:
            continue
        schema = spec.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        clean = {k: v for k, v in args.items() if k in props}
        missing = [k for k in required if k not in clean]
        if missing:
            return {"error": f"Missing required argument(s) for {name}: {', '.join(missing)}", "recoverable": True}
        try:
            return spec["fn"](**clean)
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {name}: {exc}", "recoverable": True}
    return {"error": f"Unknown tool '{name}'", "recoverable": True}


def _normalise_submission(args: dict[str, Any]):
    rec = dict(args)
    decision = str(rec.get("decision", "")).upper()
    if decision not in {"PIT", "STAY_OUT"}:
        return None, "decision must be PIT or STAY_OUT"
    if decision == "PIT" and not rec.get("compound"):
        return None, "PIT requires a compound"
    try:
        confidence = float(rec.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None, "confidence must be numeric"
    if not 0 <= confidence <= 1:
        return None, "confidence must be between 0 and 1"
    rec["decision"] = decision
    rec["confidence"] = confidence
    return rec, None


def _append_log(path: str | os.PathLike[str] | None, record: dict[str, Any]):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_agent(snapshot, question=None, verbose=True, *, backend=None, max_steps=None,
              max_tool_calls=None, log_path=None):
    """Run one decision episode. Returns ``(recommendation, trajectory)``.

    ``snapshot`` may be a RaceSnapshot or JSON-compatible dict. Every attempted
    tool call, including invalid calls and submission attempts, is logged.
    """
    snapshot = tools.set_snapshot(snapshot)
    backend = backend or create_backend()
    max_steps = max_steps or int(os.environ.get("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS))
    max_tool_calls = max_tool_calls or int(os.environ.get("AGENT_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS))
    log_path = log_path or os.environ.get("AGENT_TRAJECTORY_LOG")
    episode_id = os.environ.get("AGENT_EPISODE_ID") or f"ep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    api_tools = _api_tools()
    prompt = question or "Decide the best immediate strategy action from this snapshot."
    messages = [{"role": "user", "content": prompt + "\n\nRace snapshot:\n" + json.dumps(snapshot.to_dict(), ensure_ascii=False)}]
    trajectory = []
    tool_calls_used = 0

    if verbose:
        model_name = getattr(backend, "model", getattr(backend, "model_id", "unknown"))
        print(f"  [backend] {backend.name}: {model_name}")

    for step in range(1, max_steps + 1):
        turn = backend.run_turn(system_prompt=SYSTEM_PROMPT, messages=messages, tools=api_tools, max_tokens=4000)
        if not turn.tool_calls:
            messages.append({"role": "user", "content": "No tool call was detected. Respond with a valid tool call; submit if you have enough evidence."})
            continue

        if backend.name == "anthropic":
            messages.append({"role": "assistant", "content": turn.raw.content})
        else:
            messages.append({"role": "assistant", "content": json.dumps({"tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls]})})

        results = []
        for call in turn.tool_calls:
            if tool_calls_used >= max_tool_calls:
                messages.append({"role": "user", "content": "Tool-call budget exhausted. Submit your best recommendation now."})
                break
            tool_calls_used += 1
            args = dict(call.arguments)
            if call.name == "submit_recommendation":
                rec, error = _normalise_submission(args)
                out = {"accepted": error is None} if error is None else {"error": error, "recoverable": True}
            else:
                rec = None
                out = _dispatch(call.name, args)

            event = {"episode_id": episode_id, "step": step, "call_index": tool_calls_used, "tool": call.name,
                     "input": args, "output": out, "error": out.get("error") if isinstance(out, dict) else None}
            trajectory.append(event)
            _append_log(log_path, {**event, "timestamp": datetime.now(timezone.utc).isoformat(), "backend": backend.name})
            if verbose:
                print(f"  [tool {tool_calls_used}/{max_tool_calls}] {call.name} -> {json.dumps(out)[:240]}")

            if call.name == "submit_recommendation" and rec is not None:
                return rec, trajectory

            if backend.name == "anthropic":
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(out)})
            else:
                results.append({"tool": call.name, "result": out})

        if tool_calls_used >= max_tool_calls:
            # One final model turn is allowed to submit without additional evidence calls.
            messages.append({"role": "user", "content": "Evidence budget is exhausted. You MUST call submit_recommendation now; do not call any other tool."})
        elif backend.name == "anthropic":
            messages.append({"role": "user", "content": results})
        else:
            messages.append({"role": "user", "content": "Tool results:\n" + json.dumps(results, ensure_ascii=False)})

    return None, trajectory
