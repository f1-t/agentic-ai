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
# A turn whose calls are exactly the previous turn's (AGENT_TURN_REPEAT): "end" (default) runs
# nothing, recaps what those calls returned, and on the third identical turn in a row ends the
# evidence phase early (the grace turns follow); "remind" does the same without the early end;
# "run" runs the turn again as any other (the behaviour before the switch).
DEFAULT_TURN_REPEAT = "end"
TURN_REPEAT_MODES = ("end", "remind", "run")
TURN_REPEAT_LIMIT = 3   # repeats in a row (after the turn that ran) that end the evidence phase ("end")
RECAP_CHARS = 400       # a recap line's result, at most this long (a rule reply fits whole)
RECAP_HEAD_CHARS = 100  # its tool and arguments, at most this long

# how to read a gap between plans (AGENT_GAP_YARDSTICK "paired", the default); left out of the prompt
# under the legacy "calibration" yardstick, whose within_sim_error the tools then return
GAP_SENTENCE = ("A gap between two plans is much more certain than either plan's absolute time, because errors "
                "common to both cancel: read it from gap_reading and p_beats (the share of draws in which one plan "
                "is faster). ")

SYSTEM_PROMPT = """You are a Formula 1 race strategist making a MID-RACE decision from a snapshot in time.

The snapshot is the only current race state. The race is not named and cars are labelled (your car is Car A); decide from the snapshot and the tools, not from memory of any real race. Do not assume a fixed race, driver, lap, or starting tyre. Use tools when they reduce decision uncertainty.

IMPORTANT: tool outputs are estimates. Ranges, probabilities, confidence labels and assumptions represent uncertainty; the simulator is NOT ground truth. """ + GAP_SENTENCE + """Compare plausible outcomes and downside risk rather than blindly choosing the smallest median number.

Your task is to choose the immediate action: PIT now (and which compound) or STAY OUT for a stated number of laps. Consider position, gaps/traffic, tyre age/degradation and the estimated tyre life (a set run past it loses time quickly; there is no fixed stint limit), the pit-loss range (cheaper under a safety car or VSC), weather/track status and nearby competitors. When the tools model traffic, other cars are projected from what is known now and the tools report the expected rejoin position and time lost behind other cars. Prefer a small number of informative tool calls.

Each compound's degradation estimate carries a tyre_life field: the longest run seen so far on that compound, a pre-race tyre-life estimate with its sd, and the extra lap time charged per lap past it. It is an estimate, not a limit: wear beyond the longest run seen is uncertain, so treat stints past it as a risk to weigh, not as forbidden.

If a tool reports an invalid plan or arguments, use its error/hint to correct the call. Do not repeat the same invalid call. You have a finite step/tool budget, so submit once the evidence is sufficient.

Finish by calling submit_recommendation. Its confidence is your probability (0 to 1) that the recommended action is the better one: that pitting now beats staying out if you recommend PIT, and the reverse if you recommend STAY_OUT. Never finish with prose alone."""

SUBMIT_TOOL = {
    "name": "submit_recommendation",
    "description": "Submit the final immediate mid-race strategy decision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "description": "PIT or STAY_OUT"},
            "compound": {"type": ["string", "null"], "description": "Compound if pitting; otherwise null."},
            "reassess_after_laps": {"type": "integer", "description": "When to reassess if staying out, or after the pit."},
            "confidence": {"type": "number", "description": "0-1: the probability that the recommended action is the better one."},
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


def _turn_key(calls):
    """A turn's calls as an order-free signature: (tool, sorted-key arguments) per call."""
    return sorted((c.name, json.dumps(c.arguments, sort_keys=True, default=str)) for c in calls)


def _recap(name, args, out):
    """One line of what a call returned, for the turn-repeat message: its error (a rule reply
    says what is legal instead), its gap reading(s), or a pointer to the result above."""
    head = f"{name} {json.dumps(args, ensure_ascii=False)}" if args else name
    if not isinstance(out, dict):
        body = "returned the result shown above"
    elif out.get("error"):
        body = f"error: {out['error']}"
    elif out.get("gap_reading"):
        body = out["gap_reading"]
    elif name == "compare_plans" and (out.get("gap_readings") or any(out.get("errors") or [])):
        body = "; ".join([*(out.get("gap_readings") or []),
                          *(f"candidates[{i}] not legal: {' '.join(e)}" for i, e in enumerate(out.get("errors") or []) if e)])
    elif out.get("alternative") and out["alternative"].get("errors"):
        body = "its alternative is not legal: " + "; ".join(out["alternative"]["errors"])
    else:
        body = "returned the result shown above"
    if len(head) > RECAP_HEAD_CHARS:   # the arguments are clipped, never the result (a rule reply ends with what is legal)
        head = head[:RECAP_HEAD_CHARS - 3] + "..."
    return f"- {head}: {body if len(body) <= RECAP_CHARS else body[:RECAP_CHARS - 3] + '...'}"


def _turn_repeat_message(n, recap, ended, warn=False):
    """What the model is told about its n-th identical turn in a row: in different words the
    second time, the end of the evidence phase when `ended`, and that the next identical turn
    ends it when `warn`."""
    lines = "\n".join(recap)
    if ended:
        return ("Your turn repeated the previous one again, call for call, so the evidence phase is over and "
                f"nothing was run. What those calls returned:\n{lines}")
    if n == 1:
        return ("This turn repeats your previous one exactly (same calls, same arguments), so nothing was run "
                f"again; the results are above. What those calls returned:\n{lines}\n"
                "To move forward, make a different evidence call or call submit_recommendation.")
    return ("Again the same calls with the same arguments as before: not run, and their results have not "
            f"changed:\n{lines}\nCall a different tool or the same one with different arguments, or submit with "
            "submit_recommendation." + (" One more identical turn ends the evidence phase." if warn else ""))


def run_agent(snapshot, question=None, verbose=True, *, backend=None, max_steps=None,
              max_tool_calls=None, log_path=None, stats=None, submit_grace_steps=None,
              turn_status=None, repeat_mode=None, turn_repeat=None):
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
    a snapshot. Either way the trajectory event carries ``repeat_of``. An evidence turn whose
    calls (no submission among them) are exactly the previous turn's, as a set, is a turn
    repeat; ``turn_repeat`` (AGENT_TURN_REPEAT) says what it gets: "end" (default) runs none of
    them, uses no budget, and answers once with what they returned (_turn_repeat_message), in
    other words the second time, and the third repeat in a row (TURN_REPEAT_LIMIT) ends
    the evidence phase: the grace turns follow it at once; "remind" the same without the early
    end; "run" runs the turn like any other (repeat_mode then applies to each call). A call of
    a turn repeat is a trajectory event with ``repeat_of`` (its call in the turn that ran) and
    ``turn_repeat`` (the count) and counts in ``repeat_calls``. When ``stats`` is a
    dict it receives ``steps`` (grace turns included), ``evidence_calls`` (the evidence calls
    that ran: no submissions, no refused calls, no cached repeats; a re-run repeat counts),
    ``repeat_calls`` (the repeats that were re-run, served from the cache or answered as a turn
    repeat), ``turn_repeats`` (turn repeats not run), ``evidence_ended_at`` (the step a turn
    repeat ended the evidence phase at, None when it did not) and, if the
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
    turn_repeat = turn_repeat or os.environ.get("AGENT_TURN_REPEAT") or DEFAULT_TURN_REPEAT
    if turn_repeat not in TURN_REPEAT_MODES:
        raise ValueError(f"AGENT_TURN_REPEAT / turn_repeat must be one of {', '.join(TURN_REPEAT_MODES)}; "
                         f"got {turn_repeat!r}")
    snapshot = tools.set_snapshot(snapshot)
    stats = {} if stats is None else stats
    stats.update(steps=0, evidence_calls=0, repeat_calls=0, turn_repeats=0, evidence_ended_at=None,
                 input_tokens=None, output_tokens=None, identity=tools.identity())
    backend = backend or create_backend()
    max_steps = max_steps or int(os.environ.get("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS))
    max_tool_calls = max_tool_calls or int(os.environ.get("AGENT_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS))
    if submit_grace_steps is None:
        submit_grace_steps = int(os.environ.get("AGENT_SUBMIT_GRACE_STEPS", DEFAULT_SUBMIT_GRACE_STEPS))
    log_path = log_path or os.environ.get("AGENT_TRAJECTORY_LOG")
    episode_id = os.environ.get("AGENT_EPISODE_ID") or f"ep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    api_tools = _api_tools()
    system_prompt = (SYSTEM_PROMPT if tools.agent_tools.gap_yardstick() == "paired"
                     else SYSTEM_PROMPT.replace(GAP_SENTENCE, ""))
    prompt = question or "Decide the best immediate strategy action from this snapshot."
    messages = [{"role": "user", "content": prompt + "\n\nRace snapshot:\n" + json.dumps(snapshot.to_dict(), ensure_ascii=False)}]
    trajectory = []
    tool_calls_used = 0
    run_calls = {}   # (tool, sorted-key arguments) -> (call_index, output) of the evidence call that ran
    evidence_end = max_steps   # the last evidence turn; earlier when turn repeats end the phase
    last_turn, repeats = None, 0   # the last evidence turn that ran: (signature, {key: (call_index, output)})

    if verbose:
        model_name = getattr(backend, "model", getattr(backend, "model_id", "unknown"))
        print(f"  [backend] {backend.name}: {model_name}")

    def grace_note(step):
        # from the last evidence turn on: the turns are spent, only a submission is left
        grace_left = evidence_end + submit_grace_steps - step
        if step < evidence_end or grace_left <= 0:
            return None
        return (f"The evidence turns are spent; no further evidence calls will run. You MUST submit a "
                f"recommendation now with submit_recommendation, from the evidence above "
                f"({grace_left} submission turn{'s' if grace_left != 1 else ''} left).")

    step = 0
    while step < evidence_end + submit_grace_steps:
        step += 1
        phase = "evidence" if step <= evidence_end else "grace"
        turn = backend.run_turn(system_prompt=system_prompt, messages=messages,
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

        signature = _turn_key(turn.tool_calls)
        if (turn_repeat != "run" and phase == "evidence" and last_turn is not None and signature == last_turn[0]
                and all(c.name != "submit_recommendation" for c in turn.tool_calls)):
            # The whole turn is the previous one again: nothing runs and no budget is used. The
            # model is told so once, with what those calls returned (new information, not a bare
            # refusal), and the third time in a row ("end") the evidence phase ends.
            repeats += 1
            stats["turn_repeats"] += 1
            ended = turn_repeat == "end" and repeats >= TURN_REPEAT_LIMIT
            recap = []
            for call in turn.tool_calls:
                args = dict(call.arguments)
                key = (call.name, json.dumps(args, sort_keys=True, default=str))
                repeat_of, first = last_turn[1][key]
                recap.append(_recap(call.name, args, first))
                stats["repeat_calls"] += 1
                out = {"not_run": "repeat of the previous turn", "repeat_of": repeat_of}
                event = {"episode_id": episode_id, "step": step, "phase": phase, "call_index": len(trajectory) + 1,
                         "tool": call.name, "input": args, "output": out, "error": None,
                         "repeat_of": repeat_of, "turn_repeat": repeats}
                if getattr(turn, "parse_repair", None):
                    event["parse_repair"] = turn.parse_repair
                trajectory.append(event)
                _append_log(log_path, {**event, "timestamp": datetime.now(timezone.utc).isoformat(), "backend": backend.name})
            if ended:
                evidence_end = step
                stats["evidence_ended_at"] = step
            note = _turn_repeat_message(repeats, recap, ended,
                                        warn=turn_repeat == "end" and repeats == TURN_REPEAT_LIMIT - 1)
            if (spent := grace_note(step)):
                note += "\n\n" + spent
            if verbose:
                print(f"  [turn repeat {repeats}] {len(turn.tool_calls)} call(s) not run" + (" (evidence ended)" if ended else ""))
            if backend.name == "anthropic":
                content = [{"type": "tool_result", "tool_use_id": c.id, "content": "Not run: this call repeats your "
                            "previous turn."} for c in turn.tool_calls] + [{"type": "text", "text": note}]
            else:
                content = note
            messages.append({"role": "user", "content": content})
            continue
        repeats = 0
        ran = {}   # this turn's evidence calls: key -> (call_index, output)

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
            ran.setdefault(key, (event["call_index"], out))
            _append_log(log_path, {**event, "timestamp": datetime.now(timezone.utc).isoformat(), "backend": backend.name})
            if verbose:
                print(f"  [tool {tool_calls_used}/{max_tool_calls}] {call.name} -> {json.dumps(out)[:240]}")

            if call.name == "submit_recommendation" and rec is not None:
                return rec, trajectory

            if backend.name == "anthropic":
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(out)})
            else:
                results.append({"tool": call.name, "result": out})

        if phase == "evidence":
            last_turn = (signature, ran)
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
