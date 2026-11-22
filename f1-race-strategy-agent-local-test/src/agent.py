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

load_dotenv()

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_TOOL_CALLS = 24
DEFAULT_SUBMIT_GRACE_STEPS = 2
DEFAULT_REPEAT_MODE = "run"
REPEAT_MODES = ("run", "cache")

SYSTEM_PROMPT = """You are a Formula 1 race strategist making a MID-RACE decision from a snapshot in time.

The snapshot is the only current race state. The race is not named and cars are labelled (your car is Car A); decide from the snapshot and the tools, not from memory of any real race. Do not assume a fixed race, driver, lap, or starting tyre. Use tools when they reduce decision uncertainty.

IMPORTANT: tool outputs are estimates. Ranges, probabilities, confidence labels and assumptions represent uncertainty; the simulator is NOT ground truth. Compare plausible outcomes and downside risk rather than blindly choosing the smallest median number.

Your task is to choose the immediate action: PIT now (and which compound) or STAY OUT for a stated number of laps. Consider position, gaps/traffic, tyre age/degradation and the estimated tyre life (a set run past it loses time quickly; there is no fixed stint limit), the pit-loss range (cheaper under a safety car or VSC), weather/track status and nearby competitors. When the tools model traffic, other cars are projected from what is known now and the tools report the expected rejoin position and time lost behind other cars. Prefer a small number of informative tool calls.

Each compound's degradation estimate carries a tyre_life field: the longest run seen so far on that compound, a pre-race tyre-life estimate with its sd, and the extra lap time charged per lap past it. It is an estimate, not a limit: wear beyond the longest run seen is uncertain, so treat stints past it as a risk to weigh, not as forbidden.

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
    """The tool's output (or error) as the model is shown it: tools.present maps the real race
    and driver names to the active snapshot's pseudonyms unless AGENT_SHOW_IDENTITY is on."""
    return tools.present(_call(name, args))


def _call(name, args):
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


def _usage(raw):
    """(input_tokens, output_tokens) a backend reported for one turn, None when it did not:
    an Anthropic response's usage, or a dict payload's "usage" (either naming)."""
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is None:
        return None

    def get(*names):
        for n in names:
            v = usage.get(n) if isinstance(usage, dict) else getattr(usage, n, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return None
    pair = get("input_tokens", "prompt_tokens"), get("output_tokens", "completion_tokens")
    return None if pair == (None, None) else pair


def _finish_reason(raw):
    """Why the model stopped, as the backend reported it (an OpenAI-style choice's
    finish_reason, an Anthropic response's stop_reason), None when it did not."""
    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if choices:
        first = choices[0]
        return first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
    return raw.get("stop_reason") if isinstance(raw, dict) else getattr(raw, "stop_reason", None)


TRUNCATED = {"length", "max_tokens"}   # finish reasons of a reply cut off at the output-token cap
NO_CALL_TEXT_CHARS = 1500   # head of a turn's text logged when it had no tool call; the tail is kept too


def _clip(text, limit=NO_CALL_TEXT_CHARS):
    if len(text) <= limit:
        return text
    tail = limit // 4
    return f"{text[:limit - tail]} [... {len(text) - limit} chars ...] {text[-tail:]}"


def run_agent(snapshot, question=None, verbose=True, *, backend=None, max_steps=None,
              max_tool_calls=None, log_path=None, stats=None, submit_grace_steps=None,
              turn_status=None, repeat_mode=None):
    """Run one decision episode. Returns ``(recommendation, trajectory)``.

    ``snapshot`` may be a RaceSnapshot, a JSON-compatible dict, or a contracts.Snapshot.
    tools.set_snapshot keeps a contract for the tools and shows the model its
    RaceSnapshot.from_contract view. Every attempted tool call, including
    invalid calls, refused calls and submission attempts, is logged with its ``phase``. A turn
    with no tool call is logged too (``event`` "no_tool_call": the clipped text, the backend's
    parse error, finish reason and output tokens) but is not in the returned trajectory; the
    model is told what was wrong: a reply cut off at the output cap, or the parse error. A call
    from a reply whose JSON the backend corrected carries the backend's ``parse_repair`` note.
    With ``turn_status`` (AGENT_TURN_STATUS=1/true/on, default off) each evidence turn's
    tool-results message ends with the turn and evidence-call counts, and the second-to-last
    turn says one evidence turn is left. When the ``max_steps`` evidence turns end without an
    accepted submission, up to ``submit_grace_steps`` (AGENT_SUBMIT_GRACE_STEPS, default 2)
    grace turns follow that offer only submit_recommendation; evidence calls in them are
    refused. An evidence call identical to an earlier one this episode (same tool and
    arguments) is a repeat. ``repeat_mode`` (AGENT_REPEAT_MODE) says what it gets: "run"
    (default) runs it again against the budget, like any other call; "cache" gives it the
    cached result of the first call, error included, with ``repeat_of`` (that call_index)
    added, also once the budget is spent, and uses no budget: the tools are deterministic for
    a snapshot. Either way the trajectory event carries ``repeat_of``. When ``stats`` is a
    dict it receives ``steps`` (grace turns included), ``evidence_calls`` (the evidence calls
    that ran: no submissions, no refused calls, no cached repeats; a re-run repeat counts),
    ``repeat_calls`` (the repeats that were re-run or served from the cache) and, if the
    backend reports usage, the summed ``input_tokens`` and ``output_tokens`` (None when it
    does not), and ``identity`` (tools.identity(): the real race id and the driver labels the
    model saw in their place; the model is shown pseudonyms unless AGENT_SHOW_IDENTITY is on).
    """
    if turn_status is None:
        turn_status = os.environ.get("AGENT_TURN_STATUS", "").strip().lower() in {"1", "true", "on"}
    repeat_mode = repeat_mode or os.environ.get("AGENT_REPEAT_MODE") or DEFAULT_REPEAT_MODE
    if repeat_mode not in REPEAT_MODES:
        raise ValueError(f"AGENT_REPEAT_MODE / repeat_mode must be one of {', '.join(REPEAT_MODES)}; "
                         f"got {repeat_mode!r}")
    snapshot = tools.set_snapshot(snapshot)
    stats = {} if stats is None else stats
    stats.update(steps=0, evidence_calls=0, repeat_calls=0, input_tokens=None, output_tokens=None,
                 identity=tools.identity())
    backend = backend or create_backend()
    max_steps = max_steps or int(os.environ.get("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS))
    max_tool_calls = max_tool_calls or int(os.environ.get("AGENT_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS))
    if submit_grace_steps is None:
        submit_grace_steps = int(os.environ.get("AGENT_SUBMIT_GRACE_STEPS", DEFAULT_SUBMIT_GRACE_STEPS))
    log_path = log_path or os.environ.get("AGENT_TRAJECTORY_LOG")
    episode_id = os.environ.get("AGENT_EPISODE_ID") or f"ep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    api_tools = _api_tools()
    prompt = question or "Decide the best immediate strategy action from this snapshot."
    messages = [{"role": "user", "content": prompt + "\n\nRace snapshot:\n" + json.dumps(snapshot.to_dict(), ensure_ascii=False)}]
    trajectory = []
    tool_calls_used = 0
    run_calls = {}   # (tool, sorted-key arguments) -> (call_index, output) of the evidence call that ran

    if verbose:
        model_name = getattr(backend, "model", getattr(backend, "model_id", "unknown"))
        print(f"  [backend] {backend.name}: {model_name}")

    def grace_note(step):
        # from the last evidence turn on: the turns are spent, only a submission is left
        grace_left = max_steps + submit_grace_steps - step
        if step < max_steps or grace_left <= 0:
            return None
        return (f"The evidence turns are spent; no further evidence calls will run. You MUST submit a "
                f"recommendation now with submit_recommendation, from the evidence above "
                f"({grace_left} submission turn{'s' if grace_left != 1 else ''} left).")

    for step in range(1, max_steps + submit_grace_steps + 1):
        phase = "evidence" if step <= max_steps else "grace"
        turn = backend.run_turn(system_prompt=SYSTEM_PROMPT, messages=messages,
                                tools=api_tools if phase == "evidence" else [SUBMIT_TOOL], max_tokens=4000)
        stats["steps"] = step
        used = _usage(turn.raw)
        if used is not None:
            for key, n in zip(("input_tokens", "output_tokens"), used):
                if n is not None:
                    stats[key] = (stats[key] or 0) + n
        if not turn.tool_calls:
            # not a trajectory event (no call was attempted), but logged so the turn can be read back
            text, error, finish = turn.text or "", getattr(turn, "parse_error", None), _finish_reason(turn.raw)
            _append_log(log_path, {"episode_id": episode_id, "step": step, "phase": phase, "event": "no_tool_call",
                                   "text": _clip(text), "text_chars": len(text), "parse_error": error,
                                   "finish_reason": finish, "output_tokens": used[1] if used else None,
                                   "timestamp": datetime.now(timezone.utc).isoformat(), "backend": backend.name})
            # The model does not see its own unparsed reply, so say what was wrong with it: a bare
            # "no tool call" got the same malformed reply back, turn after turn (dev6 run 3).
            if finish in TRUNCATED:
                content = ("Your last reply was cut off at the output limit before the JSON was complete, so no "
                           "tool was run. Reply again with complete JSON and fewer tool calls in one turn.")
            elif error:
                content = (f"Your last reply was not valid JSON, so no tool was run ({error}). Reply again with "
                           f"one corrected JSON object; check that every bracket and brace is closed exactly once.")
            else:
                content = "No tool call was detected. Respond with a valid tool call; submit if you have enough evidence."
            if (spent := grace_note(step)):
                content += " " + spent
            messages.append({"role": "user", "content": content})
            continue

        if backend.name == "anthropic":
            messages.append({"role": "assistant", "content": turn.raw.content})
        else:
            messages.append({"role": "assistant", "content": json.dumps({"tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls]})})

        results = []
        for call in turn.tool_calls:
            args = dict(call.arguments)
            rec = repeat_of = None
            # A submission is always processed and never counts against the evidence budget;
            # the budget guard applies to evidence tools only. A grace turn runs no evidence. An
            # evidence call identical to one already run is run again against the budget
            # (repeat_mode "run"), or gets that call's cached result, with repeat_of, without
            # using budget, even once the budget is spent (repeat_mode "cache").
            key = (call.name, json.dumps(args, sort_keys=True, default=str))
            if call.name == "submit_recommendation":
                rec, error = _normalise_submission(args)
                out = {"accepted": error is None} if error is None else {"error": error, "recoverable": True}
            elif phase == "grace":
                out = {"error": "Evidence turns are spent; call not run. Submit your recommendation now with "
                                "submit_recommendation.", "recoverable": True}
            elif key in run_calls and repeat_mode == "cache":
                repeat_of, cached = run_calls[key]
                stats["repeat_calls"] += 1
                # a copy, so the cached output (and the first call's event) stays as it was
                out = {**cached, "repeat_of": repeat_of} if isinstance(cached, dict) else {"result": cached, "repeat_of": repeat_of}
            elif tool_calls_used >= max_tool_calls:
                out = {"error": "Tool-call budget exhausted; call not run. Submit your best recommendation now.",
                       "recoverable": True}
            else:
                if key in run_calls:   # repeat_mode "run": the repeat runs again; only the event notes it
                    repeat_of = run_calls[key][0]
                    stats["repeat_calls"] += 1
                tool_calls_used += 1
                stats["evidence_calls"] = tool_calls_used
                out = _dispatch(call.name, args)
                run_calls.setdefault(key, (len(trajectory) + 1, out))

            event = {"episode_id": episode_id, "step": step, "phase": phase, "call_index": len(trajectory) + 1,
                     "tool": call.name, "input": args, "output": out,
                     "error": out.get("error") if isinstance(out, dict) else None}
            if repeat_of is not None:
                event["repeat_of"] = repeat_of
            if getattr(turn, "parse_repair", None):   # the backend corrected the reply's JSON
                event["parse_repair"] = turn.parse_repair
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

        # Once the evidence budget is spent the turn's results are kept and the model is told to
        # submit; a later submission is still accepted, further evidence calls are refused.
        must_submit = "Evidence budget is exhausted. You MUST call submit_recommendation now; do not call any other tool."
        notes = []
        if phase == "evidence":
            if turn_status:
                notes.append(f"Turn {step} of {max_steps}. Evidence calls used {tool_calls_used} of {max_tool_calls}.")
                if step == max_steps - 1:
                    notes.append("One evidence turn is left after this one.")
            if tool_calls_used >= max_tool_calls:
                notes.append(must_submit)
        if (spent := grace_note(step)):
            notes.append(spent)
        note = " ".join(notes)
        if backend.name == "anthropic":
            content = results + ([{"type": "text", "text": note}] if note else [])
            messages.append({"role": "user", "content": content})
        else:
            content = "Tool results:\n" + json.dumps(results, ensure_ascii=False)
            if note:
                content += "\n\n" + note
            messages.append({"role": "user", "content": content})

    return None, trajectory
