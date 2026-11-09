"""Print one agent episode as a clean terminal transcript (for report screenshots).

    python scripts/show_case.py --replay reports/eval/case_b/run_01.json [--trajectory P] [--commit SHA]
    python scripts/show_case.py --live [--driver HAM --lap 30]      # calls the paid model

Run from the f1-race-strategy-agent-local-test/ directory. Standard library only (live mode imports src).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 98
ADJ = {"Spain": "Spanish", "Monaco": "Monaco", "Italy": "Italian", "Britain": "British"}
LETTER = {"MEDIUM": "M", "SOFT": "S", "HARD": "H", "INTERMEDIATE": "I", "INTER": "I", "WET": "W"}
USE_COLOR = False


def bold_red(s):
    return f"\033[1;31m{s}\033[0m" if USE_COLOR else s


def parse(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def fnum(x, fmt=".2f", default="?"):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return default


def plan_short(p):
    p = parse(p)
    try:
        return " > ".join(f"{LETTER.get(str(c).upper(), str(c)[:1])}{n}" for c, n in p)
    except (TypeError, ValueError):
        return "-"


def wrap(text, indent="", first=None, width=WIDTH):
    return textwrap.fill(str(text), width=width, initial_indent=indent if first is None else first,
                         subsequent_indent=indent, break_long_words=True)


def get_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_dotenv_quiet():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


# ---- tool call formatting -------------------------------------------------------------------
def action_label(a):
    a = parse(a)
    if not isinstance(a, dict):
        return str(a)
    kind = a.get("action") or a.get("decision") or "?"
    if kind in ("stay_out", "STAY_OUT"):
        return f"stay {a['laps']}" if "laps" in a else "stay"
    if a.get("compound"):
        return f"{kind} {a['compound']}"
    return str(kind)


def short_args(tool, inp):
    inp = parse(inp)
    if not isinstance(inp, dict) or not inp:
        return ""
    if tool == "evaluate_strategy":
        return action_label(inp.get("action"))
    if tool == "compare_plans":
        c = inp.get("candidates") or []
        return f"{len(c)} options"
    if tool == "submit_recommendation":
        parts = [str(inp.get("decision", "?"))]
        if inp.get("compound"):
            parts.append(str(inp["compound"]))
        if inp.get("reassess_after_laps") is not None:
            parts.append(f"reassess {inp['reassess_after_laps']}")
        return ", ".join(parts)
    out = []
    for k, v in inp.items():
        if isinstance(v, dict):
            v = ",".join(f"{a}:{b}" for a, b in v.items())
        out.append(f"{k}={v}")
    return ", ".join(out)


def summarize(d):
    """Return (lines, highlight_line_indexes) of key outputs for one tool call."""
    tool, inp, o = d.get("tool"), parse(d.get("input")), parse(d.get("output"))
    if d.get("error"):
        return [f"error: {d['error']}"], set()
    if isinstance(o, dict) and o.get("error") and tool != "compare_plans":
        return [f"error: {o['error']}"], set()
    if not isinstance(o, dict):
        return ([f"output: {o}"] if o else []), set()
    if tool == "estimate_tyre_degradation":
        w = (o.get("wear_s_per_lap") or {}).get("value")
        r = o.get("deg_per_lap_s") or {}
        if w is None and "median" in r:
            w = r["median"]
        if not o.get("modelled", True):
            return [f"not modelled ({o.get('reason')})"], set()
        return [f"wear median {fnum(w, '.3f')} s/lap, p05-p95 {fnum(r.get('low'), '.3f')} to "
                f"{fnum(r.get('high'), '.3f')}"], set()
    if tool == "estimate_pit_loss":
        p = o.get("pit_loss_s") or {}
        return [f"pit loss median {fnum(p.get('median'))} s, range {fnum(p.get('low'))} to "
                f"{fnum(p.get('high'))} s"], set()
    if tool == "get_model_report":
        flags = set(o.get("flags") or [])
        for v in (o.get("compounds") or {}).values():
            flags.update((v or {}).get("flags") or [])
        return [("flags: " + ", ".join(sorted(map(str, flags)))) if flags else "no flags"], set()
    if tool == "evaluate_strategy":
        l1 = (f"gap {fnum(o.get('gap_mean_s'), '+.2f')} ± {fnum(o.get('gap_std_s'))} s, "
              f"P(improves) {fnum(o.get('probability_improves_outcome'))}")
        lines, hl = [l1], set()
        if o.get("within_sim_error") is not None:
            lines.append(f"within_sim_error = {str(bool(o['within_sim_error'])).upper()} "
                         f"(basis: {o.get('confidence_basis', '?')}, calibration error "
                         f"{fnum(o.get('calibration_error_s'), '.1f')} s)")
            if o["within_sim_error"]:
                hl.add(1)
        return lines, hl
    if tool == "compare_plans":
        cands = (inp or {}).get("candidates") or [] if isinstance(inp, dict) else []
        valid, errs = o.get("valid") or [], o.get("errors") or []
        if valid and not any(valid):
            first = next((e[0] for e in errs if e), "invalid")
            return [wrap(f"all {len(valid)} options invalid; e.g. {first}", "", width=WIDTH - 6)], set()
        pb, dom = o.get("p_best") or [], o.get("dominated") or []
        items = []
        for i, c in enumerate(cands):
            s = f"{action_label(c)}: p_best {fnum(pb[i] if i < len(pb) else None)}"
            if i < len(dom) and dom[i]:
                s += ", dominated"
            items.append(s)
        return ([wrap("; ".join(items), "", width=WIDTH - 6)] if items else []), set()
    if tool == "submit_recommendation":
        return [f"accepted = {o.get('accepted')}"], set()
    return [wrap(json.dumps(o, default=str)[:160], "", width=WIDTH - 6)], set()


def print_calls(calls):
    print("Tool calls")
    print("-" * WIDTH)
    for n, d in enumerate(calls, 1):
        print(f"{n:>2}. {d.get('tool', '?')}({short_args(d.get('tool'), d.get('input'))})")
        lines, hl = summarize(d)
        for i, ln in enumerate(lines):
            for sub in ln.split("\n"):
                s = "      " + sub
                print(bold_red(s) if i in hl else s)


# ---- sections --------------------------------------------------------------------------------
def print_header(res, args, commit, replay):
    v = res.get("view") or {}
    snap = res.get("snapshot") or {}
    race = (snap.get("race") or {}).get("race_id", "")
    ev = str(v.get("event", "")).capitalize()
    race_name = f"{v.get('season', '')} {ADJ.get(ev, ev)} GP".strip()
    print(f"Race: {race_name}   Driver: {v.get('driver', '?')}   lap {v.get('lap', '?')} of "
          f"{v.get('total_laps', '?')}")
    near = {c.get("driver"): c for c in v.get("nearby_cars") or []}
    ahead = next((c["driver"] for c in near.values() if (c.get("gap_s") or 0) < 0), None)
    behind = next((c["driver"] for c in near.values() if (c.get("gap_s") or 0) > 0), None)
    ahead = ahead or snap.get("car_ahead")
    print(f"P{v.get('position', '?')}   {v.get('compound', '?')} tyres, age {v.get('tyre_age_laps', '?')} laps   "
          f"track {str(v.get('track_status', '?')).lower()}")
    print(f"Gap ahead {fnum(v.get('gap_ahead_s'), '.1f')} s ({ahead or '-'}), "
          f"gap behind {fnum(v.get('gap_behind_s'), '.1f')} s ({behind or '-'})")
    model = os.environ.get("VOYAGER_MODEL", "qwen3-30b-a3b")
    temp = os.environ.get("VOYAGER_TEMPERATURE", "1e-9")
    if args.commit:
        c = f"{args.commit} (run produced at)"
    elif replay:
        c = f"{commit} (replaying from)"
    else:
        c = commit
    print(wrap(f"model {model} | temperature {temp} | commit {c}", width=WIDTH))
    print("=" * WIDTH)


def print_submission(res, calls):
    print()
    print("Submission")
    print("-" * WIDTH)
    rec = res.get("agent_recommendation")
    if not rec:
        print(f"No submission: reached the step limit after {len(calls)} tool calls.")
        return
    print(f"Action:      {rec.get('decision', '?')}"
          + (f" ({rec['compound']})" if rec.get("compound") else ""))
    if rec.get("reassess_after_laps") is not None:
        print(f"Horizon:     reassess after {rec['reassess_after_laps']} laps")
    print(f"Confidence:  {fnum(rec.get('confidence'))}")
    label = "Rationale:   "
    print(textwrap.fill(str(rec.get("rationale", "")), width=WIDTH, initial_indent=label,
                        subsequent_indent=" " * len(label)))


def print_table(res):
    print()
    print("Scored on the world simulator")
    print("-" * WIDTH)
    ref = res.get("optimizer_reference") or {}
    if ref:
        print(f"Optimizer reference: {plan_short(ref.get('plan'))}, {ref.get('num_stops', '?')} stop, "
              f"first pit lap {ref.get('first_pit_lap', '?')}, {fnum(ref.get('time_s'), '.1f')} s")
    rows = sorted(res.get("results") or [], key=lambda r: (r.get("status") != "ok",
                  r.get("regret_s") if isinstance(r.get("regret_s"), (int, float)) and r.get("regret_s") == r.get("regret_s") else 1e18))
    hdr = f"  {'policy':<18}{'plan':<22}{'stops':>6}{'1st pit':>9}{'remaining s':>13}{'regret s':>11}"
    print(hdr)
    for r in rows:
        ok = r.get("status", "ok") == "ok"
        regret = fnum(r.get("regret_s")) if ok else str(r.get("status"))
        mark = "*" if r.get("policy") == "agent" else " "
        print(f"{mark} {r.get('policy', '?'):<18}{plan_short(r.get('plan')):<22}"
              f"{str(r.get('num_stops', '-')):>6}{str(r.get('first_pit_lap', '-')):>9}"
              f"{fnum(r.get('time_s'), '.1f', '-'):>13}{regret:>11}")


def show(res, calls, args, commit, replay):
    print_header(res, args, commit, replay)
    print_calls(calls)
    print_submission(res, calls)
    print_table(res)
    print("\n* = agent")


def read_calls(path):
    calls = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    calls.append(json.loads(line))
                except ValueError:
                    continue
    except OSError as e:
        print(f"warning: cannot read trajectory {path}: {e}", file=sys.stderr)
    calls.sort(key=lambda d: d.get("call_index", 0))
    return calls


def derive_trajectory(result_path):
    p = Path(result_path)
    m = re.match(r"(.+)\.json$", p.name)
    return p.parent.parent / "trajectories" / p.parent.name / f"{m.group(1) if m else p.stem}.jsonl"


def main(argv=None):
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Print one agent episode as a clean terminal transcript.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", metavar="RESULT_JSON", help="Replay a saved run, e.g. reports/eval/case_b/run_01.json.")
    mode.add_argument("--live", action="store_true", help="Run the agent once (calls the model; needs LLM_BACKEND=voyager).")
    ap.add_argument("--trajectory", help="Trajectory .jsonl (default: derived from the result path).")
    ap.add_argument("--commit", help="Commit SHA the run was produced at (overrides the local HEAD label).")
    ap.add_argument("--driver", default="HAM", help="Live mode: driver code (default HAM).")
    ap.add_argument("--lap", type=int, default=30, help="Live mode: laps completed (default 30).")
    ap.add_argument("--race", default="2021_04_spain", help="Live mode: race id.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    args = ap.parse_args(argv)

    USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ and not args.no_color
    load_dotenv_quiet()
    commit = get_commit()

    if args.replay:
        rp = Path(args.replay)
        res = json.loads(rp.read_text(encoding="utf-8"))
        calls = read_calls(args.trajectory or derive_trajectory(rp))
        show(res, calls, args, commit, replay=True)
        return 0

    if os.environ.get("LLM_BACKEND", "").lower() != "voyager":
        print("error: --live needs LLM_BACKEND=voyager (set it in .env).", file=sys.stderr)
        return 2
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports" / "eval" / "case_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj, out_json = out_dir / f"live_{ts}.jsonl", out_dir / f"live_{ts}.json"
    os.environ["AGENT_TRAJECTORY_LOG"] = str(traj)
    from src.baselines.runner import run_pipeline
    res = run_pipeline(race_id=args.race, driver=args.driver, lap=args.lap, verbose=False)
    out_json.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    res = json.loads(json.dumps(res, default=str))
    show(res, read_calls(traj), args, commit, replay=False)
    print(f"\nSaved: {out_json.relative_to(ROOT)} and {traj.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
