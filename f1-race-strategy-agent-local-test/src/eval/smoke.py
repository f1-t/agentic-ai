"""The rerun's smoke command: a handful of development scenarios through every stage with the real
agent, then a lint of the trajectories, rows and provenance that fails loudly (docs/eval.md "Smoke
run"). Run it after every merge that touches the agent, its tools or the belief:

    python -m src.eval.rerun smoke [--n 8] [--jobs 4] [--out-dir DIR]

The scenarios: set `smoke` of configs/eval/rerun_sets.yaml (development races only; a held-out race
of data/eval_set.json is refused, and --final never applies), sampled by the leak-free builder, then
a fixed seeded handful (pick): a lap-0 snapshot, one under a safety car (a VSC when no SC point was
sampled), one late in the race with few laps left, one mid-race, the rest spread over the other
races. Then references, baselines, the agent (--jobs workers) and the tables for just those, under
--out-dir (default reports/eval/rerun/smoke/<UTC time>, git-ignored).

The lint (lint) gives one line per finding: an episode that did not submit; a tool result with a
Python exception text, NaN or inf, a null where the tool contract promises a number, or a promised
field missing (the pit-loss estimate and track status, the tyre-life estimate with its sd, the gap
reading); a string shown to the model that names the real season, event, circuit or a driver code
(the future-blindness harness's identity check, tests/future_blindness.py); a rule reply that does
not say what is legal instead; an error row, an invalid agent row, a negative regret; provenance
without the model id, the commit, the configs or the loop switches. The command prints a ten-line
summary and the findings, and exits 1 on any finding, 0 when clean.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SET = "smoke"
LATE_LAPS = 8   # "late, few laps left": at most this many laps remaining
NUMBER = (int, float)
EXCEPTION_RE = re.compile(r"Traceback \(most recent call last\)|\b(KeyError|TypeError|ValueError|AttributeError|"
                          r"IndexError|ZeroDivisionError|AssertionError|NameError|RuntimeError|NotImplementedError|"
                          r"UnboundLocalError|ValidationError)\b|validation errors? for \w+")
NONFINITE_STR = {"nan", "-nan", "inf", "-inf", "infinity", "-infinity"}
RULE_RE = re.compile(r"minimum stint|lap minimum|Single compound|Lap count|First stint|two different compounds|"
                     r"stint limit|\bR[1-4]\b", re.IGNORECASE)
FIX_RE = re.compile(r"\blegal\b|must sum to exactly|Make it at least|drop one of the stops|continues the current set "
                    r"\([A-Z]+\)", re.IGNORECASE)
LOOP_KEYS = ("AGENT_MAX_TOOL_CALLS", "AGENT_SUBMIT_GRACE_STEPS", "AGENT_REPEAT_MODE", "AGENT_TURN_STATUS",
             "AGENT_TURN_REPEAT")
MODEL_KEY = {"voyager": "VOYAGER_MODEL", "asu": "VOYAGER_MODEL", "asu-rc": "VOYAGER_MODEL",
             "anthropic": "ANTHROPIC_MODEL", "claude": "ANTHROPIC_MODEL", "huggingface": "HF_MODEL", "hf": "HF_MODEL",
             "local": "HF_MODEL"}


# --- the scenarios -----------------------------------------------------------------------------------
def _remaining(sc) -> int:
    return sc.snapshot.race.total_laps - sc.lap


def pick(items, n: int = 8, seed: int = 0) -> list:
    """`n` scenarios, seeded: a lap-0 one, one under SC (else VSC), one with at most LATE_LAPS left
    (and more than none), one mid-race (the middle third), each from a race not used yet where there
    is one; the rest one per unused race, then any. Returned in the items' order."""
    from .agent_pass import phase_of
    rng = np.random.default_rng(seed)
    order = [items[i] for i in rng.permutation(len(items))]
    wants = [lambda s: s.lap == 0,
             lambda s: s.snapshot.track_status == "sc",
             lambda s: 0 < _remaining(s) <= LATE_LAPS,
             lambda s: phase_of(s) == "mid" and s.snapshot.track_status == "green"]
    chosen, races = [], set()

    def take(ok):
        for fresh in (True, False):
            for s in order:
                if s not in chosen and ok(s) and (not fresh or s.race_id not in races):
                    chosen.append(s)
                    races.add(s.race_id)
                    return True
        return False
    for i, want in enumerate(wants):
        if len(chosen) < n and not take(want) and i == 1:   # no SC point sampled: a VSC one
            take(lambda s: s.snapshot.track_status == "vsc")
    while len(chosen) < min(n, len(items)):
        take(lambda s: True)
    keep = {s.scenario_id for s in chosen}
    return [s for s in items if s.scenario_id in keep]


def check_development(spec: dict, races) -> None:
    """Refuse a smoke set that is not a development set or names a held-out race."""
    from .scenarios import load_eval_set
    held = load_eval_set()
    held = set(held[0]) if held else set()
    bad = sorted(set(races) & held)
    if not spec.get("development") or bad:
        raise SystemExit(f"the smoke set must be development races only; held-out races named: {bad}")


# --- the lint ----------------------------------------------------------------------------------------
def _strings(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield f"{path}/{k}", str(k)
            yield from _strings(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(x, str):
        yield path, x


def _nonfinite(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _nonfinite(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _nonfinite(v, f"{path}[{i}]")
    elif isinstance(x, float) and not math.isfinite(x):
        yield path, x
    elif isinstance(x, str) and x.strip().lower() in NONFINITE_STR:
        yield path, x


def _num(x) -> bool:
    return isinstance(x, NUMBER) and not isinstance(x, bool) and math.isfinite(x)


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _rule_replies(out: dict):
    """Every rule-reply string of a tool output: its error, its errors (flat or per candidate) and its
    alternative's errors."""
    texts = []
    if isinstance(out.get("error"), str):
        texts.append(out["error"])
    for e in out.get("errors") or []:
        texts += [e] if isinstance(e, str) else [x for x in (e or []) if isinstance(x, str)]
    alt = out.get("alternative")
    if isinstance(alt, dict):
        texts += [x for x in alt.get("errors") or [] if isinstance(x, str)]
    return [t for t in dict.fromkeys(texts) if RULE_RE.search(t)]


def _tyre_life(where, entry) -> list[str]:
    out = []
    for k in ("low", "median", "high"):
        if not _num(_get(entry, "deg_per_lap_s", k)):
            out.append(f"{where}: deg_per_lap_s.{k} is not a number")
    life = entry.get("tyre_life")
    if not isinstance(life, dict):
        out.append(f"{where}: tyre_life missing")
    elif life.get("prior_life_laps") is not None and not _num(life.get("prior_sd_laps")):
        out.append(f"{where}: tyre_life has an estimate without its sd (prior_sd_laps)")
    return out


def contract_findings(tool: str, out: dict) -> list[str]:
    """The fields the tool contract promises, on an output that is not an error."""
    if not isinstance(out, dict) or out.get("error") or out.get("not_run"):
        return []
    f = []
    if tool == "estimate_pit_loss":
        for k in ("low", "median", "high"):
            if not _num(_get(out, "pit_loss_s", k)):
                f.append(f"pit_loss_s.{k} is not a number")
        if not _num(out.get("sd_s")):
            f.append("pit-loss sd_s is not a number")
        if str(out.get("track_status")).lower() not in ("green", "sc", "vsc"):
            f.append("track_status missing")
    elif tool == "estimate_tyre_degradation" and out.get("modelled"):
        f += _tyre_life(out.get("compound", "compound"), out)
    elif tool == "get_model_report":
        comps = out.get("compounds")
        if not isinstance(comps, dict) or not comps:
            f.append("compounds missing")
        for c, entry in (comps or {}).items():
            if isinstance(entry, dict) and entry.get("modelled"):
                f += _tyre_life(c, entry)
    elif tool == "get_race_snapshot":
        if str(out.get("track_status")).lower() not in ("green", "sc", "vsc"):
            f.append("track_status missing")
    elif tool == "evaluate_strategy":
        if not _num(_get(out, "remaining", "mean_s")):
            f.append("remaining.mean_s is not a number")
        alt = out.get("alternative")
        if isinstance(alt, dict) and not alt.get("errors") and alt.get("valid", True):
            if not isinstance(out.get("gap_reading"), str):
                f.append("gap_reading missing with a legal alternative")
            if not _num(out.get("probability_improves_outcome")):
                f.append("probability_improves_outcome is not a number with a legal alternative")
    elif tool == "compare_plans":
        valid = out.get("valid") or []
        for i, ok in enumerate(valid):
            if ok and not _num(_get((out.get("summaries") or [None] * len(valid))[i] or {}, "mean_s")):
                f.append(f"summaries[{i}].mean_s is not a number for a valid candidate")
        if sum(bool(v) for v in valid) >= 2 and "gap_readings" in out:
            readings = out["gap_readings"]
            if not readings or any(not isinstance(r, str) for r in readings):
                f.append("gap_readings missing or null with two valid candidates")
    return f


def lint_trajectory(name: str, events: list[dict], identity=None) -> list[str]:
    """Findings of one episode's trajectory: exception text, NaN/inf, the contract's fields, rule
    replies without a legal alternative, no accepted submission, and (`identity`, a function of a
    value giving (path, token) hits) real names in anything the tools showed."""
    found = []
    submitted = False
    for e in events:
        tool, out = e.get("tool"), e.get("output")
        if tool is None:
            continue
        where = f"{name} call {e.get('call_index')} {tool}"
        if tool == "submit_recommendation":
            submitted = submitted or bool(isinstance(out, dict) and out.get("accepted"))
        texts = list(_strings(out)) + ([("/error", e["error"])] if isinstance(e.get("error"), str) else [])
        for path, text in texts:
            if EXCEPTION_RE.search(text):
                found.append(f"{where}: exception text at {path}: {text[:120]!r}")
        for path, v in _nonfinite(out):
            found.append(f"{where}: non-finite value {v!r} at {path}")
        found += [f"{where}: {x}" for x in contract_findings(tool, out)]
        if isinstance(out, dict):
            for text in _rule_replies(out):
                if not FIX_RE.search(text):
                    found.append(f"{where}: rule reply without a legal alternative: {text[:160]!r}")
        if identity is not None:
            found += [f"{where}: real identity {tok!r} shown at {path}" for path, tok in identity(out)]
    if not submitted:
        found.append(f"{name}: the episode did not submit")
    return found


def lint_rows(agent: pd.DataFrame, combined: pd.DataFrame | None = None) -> list[str]:
    """Error rows, invalid agent rows, agent rows without a decision, negative regret (every policy)."""
    f = []
    for _, r in agent.iterrows():
        sid = r["scenario_id"]
        if r["status"] == "error":
            f.append(f"{sid}: agent error row: {str(r.get('reason'))[:160]}")
        elif r["status"] == "invalid":
            f.append(f"{sid}: invalid agent row: {str(r.get('reason'))[:160]}")
        if r["status"] != "error" and (pd.isna(r.get("decision")) if "decision" in r else True):
            f.append(f"{sid}: no decision on the agent row")
    rows = combined if combined is not None else agent
    if combined is not None:
        for _, r in combined[(combined["status"] == "error")].iterrows():
            if r["policy"] != "agent":
                f.append(f"{r['scenario_id']}: {r['policy']} error row: {str(r.get('reason'))[:160]}")
    if "regret_s" in rows:
        neg = rows[pd.to_numeric(rows["regret_s"], errors="coerce") < -1e-6]
        f += [f"{r['scenario_id']}: negative regret {r['regret_s']:.3f} s ({r['policy']})" for _, r in neg.iterrows()]
    return f


def lint_provenance(prov: dict) -> list[str]:
    f = []
    model = prov.get("model") or {}
    backend = (model.get("LLM_BACKEND") or "anthropic").strip().lower()
    key = MODEL_KEY.get(backend)
    if key is None or not model.get(key):
        f.append(f"provenance: no model id ({key or 'unknown backend ' + backend})")
    if not prov.get("commit"):
        f.append("provenance: no commit")
    for side in ("world", "belief"):
        if not _get(prov, side, "config_hash"):
            f.append(f"provenance: no {side} config hash")
    loop = prov.get("loop")
    if not isinstance(loop, dict) or "max_steps" not in loop or any(k not in loop for k in LOOP_KEYS):
        f.append(f"provenance: loop switches missing ({', '.join(k for k in LOOP_KEYS if k not in (loop or {}))})")
    return f


def _identity_checker(race_id, profile):
    """value -> [(path, token)] naming race_id's real season, event or drivers (the harness's check)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    import future_blindness as fb
    from ..sim import jobs
    frame = jobs.load_race(race_id, profile).frame
    return lambda value: fb.identity_hits(value, race_id, frame)


def lint(d: Path, profile=None) -> list[str]:
    """Every finding under out dir `d`: each scenario's trajectory and its shown snapshot, the rows and
    provenance.json."""
    from .. import tools
    from ..sim.compute import load_profile
    from .scenarios import ScenarioSet
    d = Path(d)
    profile = profile or load_profile()
    s = ScenarioSet.load(d / "scenarios.json")
    found, checkers = [], {}
    ran = set(pd.read_csv(d / "agent.csv")["scenario_id"]) if (d / "agent.csv").exists() else None
    for sc in s.items:
        name = sc.scenario_id
        if ran is not None and name not in ran:   # an --only pass: the scenarios it ran
            continue
        path = d / "trajectories" / f"{name.replace(':', '_')}.jsonl"
        if sc.race_id not in checkers:
            checkers[sc.race_id] = _identity_checker(sc.race_id, profile)
        check = checkers[sc.race_id]
        shown = tools.set_snapshot(sc.snapshot).to_dict()
        found += [f"{name} snapshot: real identity {tok!r} shown at {p}" for p, tok in check(shown)]
        if not path.exists():
            found.append(f"{name}: no trajectory")
            continue
        events = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        found += lint_trajectory(name, events, check)
    agent = pd.read_csv(d / "agent.csv") if (d / "agent.csv").exists() else None
    combined = pd.read_csv(d / "tables" / "combined.csv") if (d / "tables" / "combined.csv").exists() else None
    if agent is None:
        found.append("no agent.csv")
    else:
        found += lint_rows(agent, combined)
    prov = d / "provenance.json"
    found += lint_provenance(json.loads(prov.read_text())) if prov.exists() else ["no provenance.json"]
    return found


# --- the summary ------------------------------------------------------------------------------------------
def summary(d: Path, findings: list[str], timings: dict, jobs: int) -> list[str]:
    """Ten lines."""
    d = Path(d)
    a = pd.read_csv(d / "agent.csv")
    a = a[a["policy"] == "agent"]
    n = len(a)
    dec = a["decision"] if "decision" in a else pd.Series([None] * n)
    grace = a["submitted_in_grace"].astype(str).str.lower().eq("true") if "submitted_in_grace" in a else pd.Series(False)
    sub = int(dec.notna().sum())

    def mean(col):
        return float(pd.to_numeric(a[col], errors="coerce").mean()) if col in a else float("nan")
    lat = pd.to_numeric(a["latency_s"], errors="coerce")
    status = ", ".join(f"{k} {v}" for k, v in a["status"].value_counts().items())
    return [f"smoke: {n} scenarios over {a['race_id'].nunique()} races -> {d}",
            f"submitted {sub}/{n} (voluntary {int((dec.notna() & ~grace).sum())}, grace {int((dec.notna() & grace).sum())})",
            f"decisions: PIT {int((dec == 'PIT').sum())}, STAY_OUT {int((dec == 'STAY_OUT').sum())}",
            f"tool calls per episode {mean('tool_calls'):.1f} (evidence {mean('evidence_calls'):.1f}, "
            f"repeats {mean('repeat_calls'):.1f})",
            f"tokens per episode: input {mean('input_tokens'):.0f}, output {mean('output_tokens'):.0f}",
            f"latency per episode: mean {lat.mean():.1f} s, median {lat.median():.1f} s (--jobs {jobs})",
            f"agent rows: {status}; api retries {int(pd.to_numeric(a.get('api_retries'), errors='coerce').fillna(0).sum()) if 'api_retries' in a else 0}",
            "stages: " + ", ".join(f"{k} {v:.0f} s" for k, v in timings.items()),
            f"total {sum(timings.values()):.0f} s",
            f"findings: {len(findings)}" + ("" if findings else " (clean)")]


def main(argv=None) -> int:
    """python -m src.eval.smoke DIR [DIR ...]: the lint over any agent pass's out dir (scenarios.json,
    trajectories/, agent.csv, tables/combined.csv, provenance.json); one FINDING line each, exit 1 on
    any."""
    dirs = sys.argv[1:] if argv is None else list(argv)
    if not dirs:
        raise SystemExit("usage: python -m src.eval.smoke DIR [DIR ...]")
    found = [f"{d}: {f}" for d in dirs for f in lint(Path(d))]
    print("\n".join(f"FINDING {f}" for f in found) or "lint: clean", flush=True)
    print(f"findings: {len(found)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
