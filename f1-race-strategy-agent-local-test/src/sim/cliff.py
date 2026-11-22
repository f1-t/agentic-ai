"""Tyre life as physics: the wear cliff (docs/eval.md "Information contract" and "The wear cliff").

A degradation model's wear is a straight line (or a line with a hinge), so with nothing else an
optimizer happily plans a 59-lap SOFT stint. The old answer, R8, forbade any stint past the
oldest tyre age run in the whole race: a rule computed from laps after the snapshot and handed
to every policy. It is retired (docs/rules.md R8). In its place each simulator prices old tyres
past the edge of what is known about them (the support edge E_c):

    lap time = the model's lap time (as fitted, inside the support)
               + min(kappa * max(0, age - E_c)^2, cap)

so a lap 2 past the edge costs kappa x 4 (0.2 s at kappa 0.05) and a dead tyre at most `cap`
(4 s/lap) more. Nothing is forbidden; an over-long stint is slow. SimConfig.wear_cliff configures
it; `enabled: false` removes it (the legacy R8 setting turns it off, docs/rules.md R8). The data
cannot locate a cliff by itself (fitted curvature is significantly negative on several
race/compound pairs: long runs are survivorship-selected), so past the support the model's
knowledge ends and this stated extrapolation takes over; kappa anywhere in 0.02-0.10 gives the
same lap-0 stop counts on the development races (docs/eval.md "The wear cliff").

Where E_c comes from. Both simulators start from the SAME historical prior on tyre life
(history_prior: races strictly before this one by date, held-out races never; behind an adapter
so the standalone history module can replace the stub, see history_prior), so the predictive
simulator and the judge agree as far as they can and the judge carries as little hindsight as
it can. Each extends the prior outward only where it has lap data:

world (fit_window "full", the scorer), world_cliffs, WearCliff.world_edge, three sources with
the same cost form. Hindsight facts are measurement targets where the judge can do without them:
the oldest age actually run is how the prior's prediction is scored, not an input, in (a).
  "history_only" (a): E_c = the prior's point edge; nothing from this race.
  "history" (b, the default until the fidelity scorecard chooses between (a) and (b)):
      E_c = max(the prior's point edge, the oldest tyre age any car ran on c in this race): the
      judge has measured laps up to that age, so it charges no extrapolation cost there. That
      extension is the judge's one whole-race input here.
  "race" (c, the earlier design): E_c = the oldest age raced in this race; for a compound no car
      raced the proxy's practice-age rule (proxy.unraced_limits).
Cliff.source records the source per compound ("history:<tier>", "history:<tier> extended by race
data to age N", "race", ...), and the world's fit_report["support_edge"] carries it
(world_ledger) for the fidelity tool's hindsight ledger.

belief (any other fit window, what a policy plans on), as_of_cliffs: the same prior with its
uncertainty drawn per sample, extended by the tyre ages seen so far: this weekend's practice and
sprint ("practice") and race laps up to the fit's calibration point (every car, "race_so_far").
A sample's edge is max(seen so far, prior + sd z), z standard normal from engine.CLIFF_STREAM, so
it never sits below what has been seen and moves up as the race shows tyres lasting longer.

The stub prior (life_prior), per compound in order, each tier with a wider sd: the same circuit
in an earlier season ("circuit_history:<race>"), else the median share of race distance the
compound's oldest stint ran in earlier races at any circuit ("history_fraction"), else a
neighbour's prior times the as-of life ratio ("life_ratio_as_of:<anchor>"), else a documented
constant share of race distance ("config_fraction", the widest). Every dry compound gets one,
whatever history exists; nothing in it reads the race it is for.

The agent's tools report the belief's edge as an estimate ("longest run seen so far on this
compound: N laps; ... wear beyond that is uncertain"), never a rule.

The circuit history table (configs/sim/tyre_life_history.json) is written by
`python -m src.sim.cliff history`, which refuses held-out races before loading anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .race_ids import as_of, canonical_race_id, parse_race_id

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "tyre_life_history.json")
COMPOUND_STEPS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "compound_steps.json")
SOFTNESS = ("SOFT", "MEDIUM", "HARD")
AS_OF_SOURCES = ("race_so_far", "practice")   # tie order: the race's own laps first


@dataclass(frozen=True)
class Cliff:
    age: float           # the support edge, where the extra cost starts (the point estimate)
    kappa: float         # s/lap per lap^2: a lap `o` laps past the edge costs kappa o^2 more ...
    cap: float           # ... at most `cap` s/lap
    source: str | None   # where `age` came from (module docstring)
    lower: float = 0.0   # belief: the oldest age seen so far, a lower bound on the cliff age
    prior: float | None = None   # belief: the pre-race prior age when it is above `lower`
    prior_sd: float = 0.0        # its sd: a sample's age is max(lower, prior + prior_sd z)
    seen_source: str | None = None   # where `lower` came from

    def extra(self, ages, edge):
        """s/lap added at `ages` with the edge at `edge`: min(kappa max(0, age - edge)^2, cap)."""
        over = np.maximum(0.0, np.asarray(ages, dtype=float) - edge)
        return np.minimum(self.kappa * over * over, self.cap)

    def sample_ages(self, z):
        """Cliff ages for standard normals z (one per sample)."""
        if self.prior is None or self.prior_sd <= 0:
            return np.full(len(z), self.age)
        return np.maximum(self.lower, self.prior + self.prior_sd * np.asarray(z))


def oldest_ages(frame, upto_lap=None) -> dict[str, float]:
    """compound -> the oldest tyre age in `frame` (every row, clean or not); with `upto_lap`
    only race rows with lap <= upto_lap."""
    if frame is None or len(frame) == 0 or "compound" not in frame or "tyre_age" not in frame:
        return {}
    f = frame[frame["compound"].notna() & frame["tyre_age"].notna()]
    if upto_lap is not None:
        f = f[f["lap"].notna() & (f["lap"] <= upto_lap)]
    return {str(c): float(a) for c, a in f.groupby("compound")["tyre_age"].max().items()}


def world_cliffs(laps, practice, race, wc, steps=None) -> dict[str, Cliff]:
    """The scorer's cliffs (module docstring, world) for `race` (RaceParams), by
    WearCliff.world_edge. Every compound gets its edge, a fitted hinge or not; Cliff.source says
    where it came from (the judge's whole-race input, for the fidelity ledger). `steps` is the
    proxy's step table, None when the compound proxy is off."""
    raced = oldest_ages(laps)
    if wc.world_edge == "race":
        from . import proxy
        ages, sources = raced, {}
        if steps is not None:
            ages, sources = proxy.unraced_limits(ages, practice, steps)
        return {c: Cliff(a, wc.kappa, wc.cap_s_per_lap, sources.get(c, "race")) for c, a in sorted(ages.items())}
    out = {}
    prior = history_prior(race.race_id, race.total_laps, wc, role="world")
    if wc.world_edge == "history_only":
        raced = {c: a for c, a in raced.items() if c not in prior}   # only where there is no prior
    for c in sorted(set(prior) | set(raced)):
        p = prior.get(c)
        seen = raced.get(c)
        if p is None:   # no prior (a wet compound): the race's own support
            out[c] = Cliff(seen, wc.kappa, wc.cap_s_per_lap, "race (no history prior)")
        elif seen is not None and seen > p[0]:
            out[c] = Cliff(seen, wc.kappa, wc.cap_s_per_lap,
                           f"history:{p[2]} extended by race data to age {seen:g}")
        else:
            out[c] = Cliff(p[0], wc.kappa, wc.cap_s_per_lap, f"history:{p[2]}")
    return out


def world_ledger(cliffs: dict) -> dict[str, dict]:
    """compound -> {edge, source, whole_race_input}: the judge's edges and whether each used this
    race's laps (a whole-race input), for the fidelity ledger."""
    def whole_race(src):
        return not str(src).startswith("history:") or "extended by race data" in str(src)
    return {c: {"edge": cl.age, "source": cl.source, "whole_race_input": whole_race(cl.source)}
            for c, cl in cliffs.items()}


def history_prior(race_id, total_laps, wc, role="belief") -> dict[str, tuple[float, float, str]]:
    """compound -> (edge laps, sd laps, tier): the historical tyre-life prior both simulators
    share (`role` is "world" or "belief"; the prior is the same for both). WearCliff.prior_source
    picks it: "tyre_history" (the default) is src/sim/tyre_history.py, tyre_life_prior(race_id,
    compound, total_laps) (C-number and circuit weighted, its own fallback chain, tiers 1-4);
    "stub" is this module's as-of life_prior. Both are strictly as-of by date and never read a
    held-out race. A race id with no date (a synthetic race) takes the stub, whose chain ends in
    a documented constant."""
    rid, _ = _race_key(race_id)
    if wc.prior_source == "stub" or rid is None:
        return life_prior(race_id, total_laps, wc)
    from .tyre_history import tyre_life_prior
    out = {}
    for c in SOFTNESS:
        p = tyre_life_prior(rid, c, total_laps=total_laps)
        out[c] = (float(p.edge_laps), float(p.sd_laps), f"tyre_history_tier{p.tier}")
    return out


def _race_key(race_id):
    try:
        return canonical_race_id(race_id), parse_race_id(race_id)
    except (ValueError, TypeError):
        return None, None


def as_of_life_ratio(race_id, rows, soft, hard) -> float | None:
    """The median life_ratio (oldest age softer / harder) of the pair over `rows` (the per-race
    rows of configs/sim/compound_steps.json, thin rows and NaN ratios left out) of races strictly
    before race_id; None when there are none or the id does not parse."""
    rid, _ = _race_key(race_id)
    if rid is None:
        return None
    key = f"{soft}-{hard}"
    use = [r for r in rows or [] if r.get("pair") == key and not r.get("thin")
           and r.get("life_ratio") is not None and math.isfinite(float(r["life_ratio"]))]
    earlier = set(as_of(rid, sorted({r["race_id"] for r in use})))
    vals = [float(r["life_ratio"]) for r in use if r["race_id"] in earlier]
    return float(np.median(vals)) if vals else None


def _ratio_between(race_id, rows, anchor, c):
    """Factor taking anchor's age to c's along the neighbour pairs (as-of), None if a step is missing."""
    i, j = SOFTNESS.index(anchor), SOFTNESS.index(c)
    g = 1.0
    for k in range(min(i, j), max(i, j)):
        r = as_of_life_ratio(race_id, rows, SOFTNESS[k], SOFTNESS[k + 1])
        if r is None or r <= 0:
            return None
        g *= r if j < i else 1.0 / r
    return g


PRIOR_TIERS = ("circuit_history", "history_fraction", "life_ratio_as_of", "config_fraction")


def _scan_candidates(race_id) -> list[str]:
    """Cached races (real_run.cached_races, by date from the cached schedules) strictly before
    race_id, held-out ones left out (cross_race.split_held_out reads data/eval_set.json only, so
    nothing about a held-out race is loaded)."""
    from .cross_race import split_held_out
    from .real_run import cached_races
    rid, _ = _race_key(race_id)
    if rid is None:
        return []
    if "ids" not in _CACHED:   # once per process
        _CACHED["ids"] = split_held_out(cached_races()[0])[0]
    return as_of(rid, _CACHED["ids"])


_CACHED: dict = {}


_SCANNED: dict = {}


def _scan_race(rid) -> dict | None:
    """{"circuit", "season", "round", "total_laps", "oldest_age"} of a cached race from its race
    session alone (no fit, no pit loss), memoised per process; None when it does not load."""
    if rid not in _SCANNED:
        try:
            from .calibrate import load_session
            from .data import lap_frame
            from .event_rates import circuit_info
            season, rnd, slug = parse_race_id(rid)
            session = load_session(season, rnd)
            circuit = circuit_info([rid])["circuit"].iloc[0]
            _SCANNED[rid] = {"circuit": str(circuit or slug), "season": season, "round": rnd,
                             "total_laps": int(session.total_laps), "oldest_age": oldest_ages(lap_frame(session.laps))}
        except Exception:   # noqa: BLE001  a race that does not load gives no history, never a failure
            _SCANNED[rid] = None
    return _SCANNED[rid]


def history_for(race_id, wc) -> dict:
    """{race_id: entry} of every race the prior may read for `race_id`: the committed table
    (WearCliff.history) plus, with WearCliff.scan_cache, every other cached race strictly before
    it (held-out races never), read from the cache at run time, so newly cached seasons are picked
    up without a code change. Only races strictly before race_id are used (life_prior)."""
    races = dict((wc.history or {}).get("races") or {})
    if getattr(wc, "scan_cache", False):
        for r in _scan_candidates(race_id):
            if r not in races and (e := _scan_race(r)) is not None:
                races[r] = e
    return races


def life_prior(race_id, total_laps, wc) -> dict[str, tuple[float, float, str]]:
    """compound -> (prior cliff age, its sd, source): the pre-race tyre-life prior, from races
    strictly before race_id only (race_ids.as_of, history_for), in this order, each tier with a
    wider sd than the one before (WearCliff.prior_sd_fraction[tier] x this race's laps):
      1. "circuit_history:<race>": the oldest age raced on the compound at this circuit in the
         latest earlier race there, scaled by this race's laps over that race's;
      2. "history_fraction": the median over earlier races at any circuit (races in
         WearCliff.life_excluded left out) of oldest age raced / race laps, times this race's laps;
      3. "life_ratio_as_of:<anchor>": a neighbour's prior times the as-of life ratio;
      4. "config_fraction": WearCliff.prior_life_fraction times this race's laps, the documented
         constant prior, always available.
    Every dry compound gets a prior, whatever history exists; nothing here reads this race's laps."""
    rid, parsed = _race_key(race_id)
    races = history_for(race_id, wc) if rid is not None else {}
    sd = {t: float(wc.prior_sd_fraction[t]) * total_laps for t in PRIOR_TIERS}
    out: dict[str, tuple[float, float, str]] = {}
    earlier = as_of(rid, sorted(races)) if rid is not None else []
    circuit = ((races.get(rid) or {}).get("circuit") or parsed[2]) if rid is not None else None
    usable = [r for r in earlier if r not in wc.life_excluded and races[r].get("total_laps")]
    for c in SOFTNESS:
        same = [r for r in usable if races[r].get("circuit") == circuit and c in races[r].get("oldest_age", {})]
        if same:
            r = max(same, key=lambda x: parse_race_id(x)[:2])
            out[c] = (races[r]["oldest_age"][c] * total_laps / races[r]["total_laps"], sd["circuit_history"],
                      f"circuit_history:{r}")
            continue
        fr = [races[r]["oldest_age"][c] / races[r]["total_laps"] for r in usable if c in races[r].get("oldest_age", {})]
        if fr:
            out[c] = (float(np.median(fr)) * total_laps, sd["history_fraction"], "history_fraction")
    for c in SOFTNESS:
        if c in out or not out:
            continue
        anchor = min(out, key=lambda a: (abs(SOFTNESS.index(a) - SOFTNESS.index(c)), SOFTNESS.index(a)))
        g = _ratio_between(race_id, wc.life_ratio_rows, anchor, c)
        if g is not None:
            out[c] = (out[anchor][0] * g, sd["life_ratio_as_of"], f"life_ratio_as_of:{anchor}")
    for c, f in (wc.prior_life_fraction or {}).items():
        if c not in out:
            out[c] = (float(f) * total_laps, sd["config_fraction"], "config_fraction")
    return out


def as_of_cliffs(rows, practice, race, wc) -> dict[str, Cliff]:
    """The belief's cliffs for `race` (RaceParams) from `rows`, the race rows visible at the fit's
    calibration point (Simulator.window_rows: cut by session time, the same cut as the fit;
    never an advanced snapshot's laps, which are a plan's hypothetical future), and this weekend's
    practice (module docstring, belief). Per compound the age observed so far is a lower bound,
    the historical prior (history_prior) a wide estimate: a sample's edge is
    max(lower, prior + sd z)."""
    seen: dict[str, tuple[float, str]] = {}

    def offer(c, a, src):
        if c not in seen or a > seen[c][0]:
            seen[c] = (float(a), src)
    so_far = oldest_ages(rows)
    for c, a in so_far.items():
        offer(c, a, "race_so_far")
    for c, a in oldest_ages(practice).items():
        offer(c, a, "practice")
    prior = history_prior(race.race_id, race.total_laps, wc, role="belief")
    out = {}
    for c in sorted(set(seen) | set(prior)):
        lower, lsrc = seen.get(c, (0.0, None))
        p, psd, psrc = prior.get(c, (None, 0.0, None))
        if p is None or lower >= p:
            out[c] = Cliff(lower, wc.kappa, wc.cap_s_per_lap, lsrc, lower=lower)
        else:
            out[c] = Cliff(p, wc.kappa, wc.cap_s_per_lap, psrc, lower=lower, prior=p, prior_sd=psd,
                           seen_source=lsrc)
    return out


def describe(cliff: Cliff) -> str:
    """The tools' sentence for one compound's belief cliff."""
    where = {"race_so_far": "in this race so far", "practice": "in practice this weekend"}
    seen = cliff.lower if cliff.prior is not None else cliff.age
    src = cliff.seen_source if cliff.prior is not None else cliff.source
    text = (f"longest run seen so far on this compound: {seen:g} laps ({where[src]}); "
            if src in where else "no run seen on this compound yet; ")
    if cliff.prior is not None:
        text += (f"pre-race tyre-life estimate {cliff.prior:.0f} laps (sd {cliff.prior_sd:.0f}, "
                 f"{str(cliff.source).split(':')[0]}); ")
    return text + "wear beyond that is uncertain"


# --- loaders for SimConfig.wear_cliff -----------------------------------------------------
def load_history(path=HISTORY_FILE) -> dict:
    """The committed circuit history table, {} when absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        table = json.load(f)
    return {"races": table.get("races", {})}


def load_life_ratio_rows(path=COMPOUND_STEPS_FILE) -> list[dict]:
    """The per-race life_ratio rows of configs/sim/compound_steps.json (race_id, pair, thin,
    life_ratio), [] when absent."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        table = json.load(f)
    out = []
    for r in table.get("per_race", []):
        v = r.get("life_ratio")
        out.append({"race_id": r["race_id"], "pair": r["pair"], "thin": bool(r.get("thin", False)),
                    "life_ratio": None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)})
    return out


# --- the history table --------------------------------------------------------------------
def history_rows(race_ids, loader, profile) -> dict:
    """{race_id: {"circuit", "season", "round", "total_laps", "oldest_age"}} of each race (loaded
    with `loader(race_id, profile)`)."""
    from .event_rates import circuit_info
    circuits = circuit_info(race_ids).set_index("race_id")["circuit"].to_dict()
    out = {}
    for rid in race_ids:
        data = loader(rid, profile)
        season, rnd, _ = parse_race_id(rid)
        out[rid] = {"circuit": str(circuits.get(rid, parse_race_id(rid)[2])), "season": season, "round": rnd,
                    "total_laps": int(data.race.total_laps), "oldest_age": oldest_ages(data.frame)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.sim.cliff")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history", help="write the circuit tyre-life history table from cached races")
    h.add_argument("--races", nargs="+", required=True, help="explicit race ids (never auto)")
    h.add_argument("--out", default=HISTORY_FILE)
    h.add_argument("--profile", default=None)
    args = p.parse_args(argv)
    from . import cross_race
    from .compute import load_profile
    if "auto" in args.races:
        raise SystemExit("give explicit race ids")
    cross_race.guard_held_out(args.races)   # refuses a held-out race before anything is loaded
    rows = history_rows([canonical_race_id(r) for r in args.races], cross_race.load_race, load_profile(args.profile))
    table = {"description": ("Per cached race, the oldest tyre age any car ran on each compound in the race "
                             "(every lap). The belief's wear cliff (src/sim/cliff.py) reads the races at the "
                             "same circuit strictly before the race it plans for; compound names are the "
                             "weekend's labels (SOFT/MEDIUM/HARD), not Pirelli's C-numbers."),
             "command": "python -m src.sim.cliff history --races " + " ".join(args.races),
             "held_out_never_loaded": "data/eval_set.json", "races": rows}
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(rows, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
