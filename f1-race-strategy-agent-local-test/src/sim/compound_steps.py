"""The tier-5 neighbour-step table of the compound proxy (proxy.py), and the proxy's validation.

    python -m src.sim.compound_steps estimate [--races ...] [--out configs/sim/compound_steps.json]
    python -m src.sim.compound_steps validate [--races ...] [--out-dir reports/calibration/compound_proxy]

Calibration races only: a race data/eval_set.json holds out is refused before anything about it
is loaded (cross_race.guard_held_out, never allowed here), and the default race list is
CALIBRATION_RACES. The simulator never runs this module; it reads the committed table through
SimConfig.compound_proxy.steps (config.load_compound_steps), so config_hash covers the numbers.

estimate: per race, the full-window world fit (the race's default SimConfig with the proxy off)
and the compounds with min_samples clean race rows (tier 1). For each neighbour pair (SOFT-MEDIUM,
MEDIUM-HARD) with both compounds tier 1 and at least MIN_PAIR_ROWS (50) clean race rows each:
  pace_frac   base_softer / base_harder - 1 (the fresh-tyre step as a share of the lap)
  wear_ratio  wear_softer / wear_harder, only when wear_harder > 0 (else the ratio means nothing)
  life_ratio  R8 limit_softer / limit_harder (rules.stint_limits, the oldest tyre run), not from
              races in LIFE_RATIO_EXCLUDED (2025 Qatar: a mandated 25-lap stint cap)
Each step's value is the median over the races, its sd the sample sd across them (ddof 1; NaN
with one race) and n the races. The table keeps every race's values, thin ones flagged.

validate: for each calibration race and each tier-1 compound X of it, X's race laps are hidden
(its rows dropped from the race frame), the world refitted, and X priced again by each source
alone (practice, qualifying for pace, the cross-race step; the step table without this race;
proxy.apply's `sources`), then by the blend leave-one-race-out (weights_from the other races'
source rows, the step table without this race). Against the full fit's X: fresh pace at
mid-race (lap total_laps // 2, age 0), wear, and the time of a 15- and a 25-lap stint on a new
set centred on mid-race. The shipped weights (1 / each source's mean squared error over every
race) and sd floor (the blend's leave-one-race-out root mean squared error) are written into the
table. Then the decision level, for the step alone and for the blend: from lap 0 of one finisher
per starting compound, the reference (optimizer_bridge.optimize_batch, seed 0, 50 samples, two
stops) on the world with X hidden and proxied against the one on the true world: whether the
plan changes, the true world's time of the proxied plan less the true optimum, and the laps the
proxied plan puts on X less the true plan's. Writes the csvs listed in `validate_main`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import proxy

ROOT = Path(__file__).resolve().parent.parent.parent
STEPS_OUT = ROOT / "configs" / "sim" / "compound_steps.json"
REPORT_DIR = ROOT / "reports" / "calibration" / "compound_proxy"
CALIBRATION_RACES = ("2021_04_spain", "2023_07_spain", "2023_01_bahrain", "2019_10_united_kingdom",
                     "2022_02_saudi_arabia", "2024_07_emilia_romagna", "2025_23_qatar")
LIFE_RATIO_EXCLUDED = {"2025_23_qatar": "mandated 25-lap maximum stint, so its tyre ages say nothing about tyre life"}
STINT_LAPS = (15, 25)
STEP_KEYS = ("pace_frac", "wear_ratio", "life_ratio")
MIN_PAIR_ROWS = 50   # clean race rows on both compounds of a pair before a race's step counts
COMMAND = "python -m src.sim.compound_steps estimate"
DECISION_SAMPLES, DECISION_STOPS = 50, 2

_DATA: dict = {}


def load(race_id):
    """jobs.load_race under the local profile, once per process; held-out races refused."""
    from . import cross_race, jobs
    from .compute import load_profile
    cross_race.guard_held_out([race_id])
    if race_id not in _DATA:
        _DATA[race_id] = jobs.load_race(race_id, load_profile())
    return _DATA[race_id]


def proxy_off(cfg):
    return cfg.model_copy(update={"compound_proxy": cfg.compound_proxy.model_copy(update={"enabled": False})})


def with_steps(cfg, steps):
    return cfg.model_copy(update={"compound_proxy": cfg.compound_proxy.model_copy(update={"steps": steps})})


def tier1(deg, min_samples):
    rows = deg.fit_report["n_race_rows_per_compound"]
    return [c for c in proxy.SOFTNESS if c in deg.coefs and rows.get(c, 0) >= min_samples]


# --- estimate ------------------------------------------------------------------------------------
def race_steps(race_id, data=None) -> list[dict]:
    """One row per neighbour pair with both compounds raced (tier 1) in `race_id`'s world fit."""
    from .engine import Simulator
    data = data or load(race_id)
    cfg = proxy_off(data.config)
    world = Simulator(data.frame, cfg, data.practice)
    deg = world.degradation
    raced = tier1(deg, cfg.min_samples)
    rows = []
    for soft, hard in proxy.PAIRS:
        if soft not in raced or hard not in raced:
            continue
        (bs, ws), (bh, wh) = deg.coefs[soft], deg.coefs[hard]
        ls, lh = world.stint_limits.get(soft), world.stint_limits.get(hard)
        rows.append({"race_id": race_id, "pair": proxy.pair_key(soft, hard),
                     "pace_frac": bs / bh - 1.0, "pace_step_s": bs - bh,
                     "wear_ratio": ws / wh if wh > 0 else np.nan,
                     "life_ratio": (ls / lh if race_id not in LIFE_RATIO_EXCLUDED and ls and lh else np.nan),
                     "base_soft": bs, "base_hard": bh, "wear_soft": ws, "wear_hard": wh,
                     "limit_soft": ls, "limit_hard": lh,
                     "race_rows_soft": int(deg.fit_report["n_race_rows_per_compound"][soft]),
                     "race_rows_hard": int(deg.fit_report["n_race_rows_per_compound"][hard])})
        rows[-1]["thin"] = min(rows[-1]["race_rows_soft"], rows[-1]["race_rows_hard"]) < MIN_PAIR_ROWS
    return rows


def aggregate(per_race: pd.DataFrame, exclude=(), keep_thin=False) -> dict:
    """{pair: {key: {value, sd, n, races}}}: the median, the sample sd across races and the
    count, over the rows of races not in `exclude` with a finite value; a thin row (fewer than
    MIN_PAIR_ROWS clean race rows on either compound) is left out unless `keep_thin`."""
    out = {}
    rows = per_race[~per_race["race_id"].isin(exclude)] if len(per_race) else per_race
    if len(rows) and not keep_thin and "thin" in rows:
        rows = rows[~rows["thin"].astype(bool)]
    for pair, g in rows.groupby("pair", sort=True) if len(rows) else []:
        entry = {}
        for key in STEP_KEYS:
            v = g[["race_id", key]].dropna()
            if not len(v):
                continue
            x = v[key].to_numpy(dtype=float)
            entry[key] = {"value": float(np.median(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                          "n": int(len(x)), "races": sorted(v["race_id"])}
        out[pair] = entry
    return out


def _json_num(x):
    return None if isinstance(x, float) and not np.isfinite(x) else x


def write_table(per_race: pd.DataFrame, races, path=STEPS_OUT):
    """Write the step table; the blend sections ("weights", "sd_floor", "blend_provenance"),
    which `validate` writes, are kept."""
    steps = aggregate(per_race)
    old = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    doc = {
        "description": ("Tier-5 neighbour steps of the compound proxy (src/sim/proxy.py): per pair of "
                        "neighbouring compounds, the median over calibration races of pace_frac (base_softer / "
                        "base_harder - 1), wear_ratio (wear_softer / wear_harder, races with wear_harder > 0) and "
                        "life_ratio (R8 limit_softer / limit_harder); sd is the sample sd across the races."),
        "fit": "full-window world fit per race, the race's default SimConfig with compound_proxy off; a compound "
               "counts when it has min_samples clean race rows",
        "command": COMMAND,
        "races": list(races),
        "life_ratio_excluded": LIFE_RATIO_EXCLUDED,
        "held_out_never_loaded": "data/eval_set.json",
        "steps": {pair: {k: {kk: _json_num(vv) for kk, vv in e.items()} for k, e in entry.items()}
                  for pair, entry in steps.items()},
        "min_pair_race_rows": MIN_PAIR_ROWS,
        "per_race": [{k: _json_num(v) if not isinstance(v, (str, bool, np.bool_)) else
                      (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in r.items()}
                     for r in per_race.to_dict("records")],
        **{k: old[k] for k in ("weights", "sd_floor", "blend_provenance") if k in old},
    }
    Path(path).write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return steps


def per_race_table(races) -> pd.DataFrame:
    rows = [r for rid in races for r in race_steps(rid)]
    return pd.DataFrame(rows)


def estimate_main(argv):
    ap = argparse.ArgumentParser(prog="compound_steps estimate", description="the tier-5 neighbour-step table")
    ap.add_argument("--races", nargs="+", default=list(CALIBRATION_RACES))
    ap.add_argument("--out", default=str(STEPS_OUT))
    args = ap.parse_args(argv)
    from . import cross_race
    cross_race.guard_held_out(args.races)
    per_race = per_race_table(args.races)
    steps = write_table(per_race, args.races, args.out)
    print(per_race.round(4).to_string(index=False))
    for pair, entry in steps.items():
        for k, e in entry.items():
            print(f"{pair} {k}: {e['value']:.4f} sd {e['sd']:.4f} n {e['n']}")
    print(f"written to {args.out}")


# --- validate ------------------------------------------------------------------------------------
# one source at a time (its error sets its weight), then the blend; qualifying informs pace only,
# so its case borrows the step for wear and only its pace error is scored
SOURCE_CASES = {"practice": {"pace": ["practice"], "wear": ["practice"]},
                "qualifying": {"pace": ["qualifying"], "wear": ["cross_race_step"]},
                "cross_race_step": {"pace": ["cross_race_step"], "wear": ["cross_race_step"]}}
CASES = (*SOURCE_CASES, "blend")
WEIGHT_ERRORS = {"pace": "pace_err_s", "wear": "wear_err_s_per_lap"}


def _stint(model, compound, n, total):
    mid = total // 2
    laps = np.arange(mid - n // 2, mid - n // 2 + n, dtype=float)
    return float(np.sum(model.lap_times(compound, np.arange(1, n + 1, dtype=float), laps)))


def _price_hidden(data, x, sources, steps, weights=None):
    """(model with x priced through `sources` or None when they cannot, its proxy entry)."""
    from .engine import Simulator
    cfg = proxy_off(data.config)
    hidden = data.frame[data.frame["compound"] != x]
    sim = Simulator(hidden, cfg, data.practice)
    px = with_steps(data.config, steps).compound_proxy
    if weights is not None:
        px = px.model_copy(update={"weights": weights})
    m = proxy.apply(sim.degradation, px, min_samples=cfg.min_samples, practice=sim.physics_prior(),
                    quali=proxy.qualifying_gaps(data.qualifying), sources=sources,
                    knot=lambda c: cfg.hinge_knots.get(c, cfg.hinge_knot_default))
    entry = (m.fit_report.get("proxy") or {}).get(x)
    if x not in m.coefs or entry is None or entry["pace_tier"] == 1:
        return None, None
    return m, entry


def weights_from(rows: pd.DataFrame, exclude=()) -> tuple[dict, dict]:
    """(weights, mse): per kind and source, 1 / the mean squared error of that source alone over
    the validation rows of races not in `exclude` (fresh-pace error for pace, wear error for
    wear; qualifying has no wear weight), and the mean squared errors with their n."""
    weights, mse = {"pace": {}, "wear": {}}, {"pace": {}, "wear": {}}
    g = rows[rows["available"].astype(bool) & ~rows["race_id"].isin(exclude)]
    for kind, col in WEIGHT_ERRORS.items():
        for source in SOURCE_CASES:
            v = g.loc[g["case"] == source, col].dropna()
            if kind == "wear" and source == "qualifying":
                continue
            if len(v):
                m = float((v ** 2).mean())
                mse[kind][source] = {"mse": m, "n": int(len(v))}
                weights[kind][source] = 1.0 / m
    return weights, mse


def _row(race_id, x, case, m, entry, true, total):
    row = {"race_id": race_id, "compound": x, "case": case, "available": m is not None}
    if m is not None:
        mid = total // 2
        row.update(pace_source=entry["pace_source"], wear_source=entry["wear_source"],
                   pace_weights=";".join(f"{p['source']}:{p['weight']:.3f}" for p in entry.get("pace_sources", [])),
                   wear_weights=";".join(f"{p['source']}:{p['weight']:.3f}" for p in entry.get("wear_sources", [])),
                   pace_anchor=entry["pace_anchor"], wear_anchor=entry["wear_anchor"],
                   ordering=";".join(entry["ordering"]),
                   pace_err_s=float(m.lap_times(x, [0.0], [mid])[0] - true.lap_times(x, [0.0], [mid])[0]),
                   wear_err_s_per_lap=float(m.coefs[x][1] - true.coefs[x][1]),
                   **{f"stint{n}_err_s": _stint(m, x, n, total) - _stint(true, x, n, total) for n in STINT_LAPS},
                   true_base_s=true.coefs[x][0], true_wear=true.coefs[x][1],
                   proxy_base_s=m.coefs[x][0], proxy_wear=m.coefs[x][1])
        if case == "qualifying":   # pace only: the wear and the stint times are the step's
            row.update(wear_err_s_per_lap=np.nan, **{f"stint{n}_err_s": np.nan for n in STINT_LAPS})
    return row


def source_rows(race_id, per_race) -> list[dict]:
    """Each raced compound of `race_id` hidden and priced by one source at a time (the step
    table without this race)."""
    from .engine import Simulator
    data = load(race_id)
    cfg = proxy_off(data.config)
    true = Simulator(data.frame, cfg, data.practice).degradation
    steps = aggregate(per_race, exclude=[race_id])    # leave-one-race-out
    rows = []
    for x in tier1(true, cfg.min_samples):
        for case, sources in SOURCE_CASES.items():
            m, entry = _price_hidden(data, x, sources, steps)
            rows.append(_row(race_id, x, case, m, entry, true, data.race.total_laps))
    return rows


def blend_rows(race_id, per_race, sources_table) -> list[dict]:
    """Each raced compound of `race_id` hidden and priced by the blend, leave-one-race-out: the
    weights from the other races' source rows and the step table without this race."""
    from .engine import Simulator
    data = load(race_id)
    cfg = proxy_off(data.config)
    true = Simulator(data.frame, cfg, data.practice).degradation
    steps = aggregate(per_race, exclude=[race_id])
    weights, _ = weights_from(sources_table, exclude=[race_id])
    rows = []
    for x in tier1(true, cfg.min_samples):
        m, entry = _price_hidden(data, x, list(proxy.SOURCES_ALL), steps, weights)
        rows.append(_row(race_id, x, "blend", m, entry, true, data.race.total_laps))
    return rows


ERROR_COLUMNS = ("pace_err_s", "wear_err_s_per_lap", "stint15_err_s", "stint25_err_s")


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per case: n, mean signed error, mean absolute error and root mean squared error of each
    error column."""
    out = []
    for case in CASES:
        g = rows[(rows["case"] == case) & rows["available"].astype(bool)]
        r = {"case": case, "n": int(len(g))}
        for col in ERROR_COLUMNS:
            v = g[col].dropna() if col in g else pd.Series(dtype=float)
            r[f"{col}_n"] = int(len(v))
            r[f"{col}_mean"] = float(v.mean()) if len(v) else np.nan
            r[f"{col}_mae"] = float(v.abs().mean()) if len(v) else np.nan
            r[f"{col}_rmse"] = float(np.sqrt((v ** 2).mean())) if len(v) else np.nan
        out.append(r)
    return pd.DataFrame(out)


def _laps_on(plan, x):
    return sum(s.laps for s in plan.stints if s.compound == x)


DECISION_ARMS = ("cross_race_step", "blend")


def decision_rows(race_id, per_race, sources_table, arm="blend") -> list[dict]:
    """The lap-0 references with each raced compound hidden and priced by `arm` ("blend": every
    source, leave-one-race-out weights; "cross_race_step": the step alone), against the true
    world's, one finisher per starting compound."""
    from ..baselines.domain import NoFeasiblePlan
    from . import calibration
    from .engine import Simulator
    from .optimizer_bridge import optimize_batch
    data = load(race_id)
    steps = aggregate(per_race, exclude=[race_id])
    weights, _ = weights_from(sources_table, exclude=[race_id])
    sources = list(proxy.SOURCES_ALL) if arm == "blend" else ["cross_race_step"]
    cfg = data.config.model_copy(update={"compound_proxy": data.config.compound_proxy.model_copy(
        update={"steps": steps, "weights": weights, "sources": sources})})
    true_world = Simulator(data.frame, cfg, data.practice, qualifying=data.qualifying)
    raced = tier1(true_world.degradation, cfg.min_samples)
    starters = {}
    for d in calibration._finisher_ids(data.frame, 0.9):
        snap = calibration.with_race(calibration.snapshot_at(data.frame, data.race, d, 0), data.race)
        starters.setdefault(snap.compound, snap)
    rows = []
    for x in raced:
        hidden = Simulator(data.frame[data.frame["compound"] != x], cfg, data.practice, qualifying=data.qualifying)
        entry = hidden.degradation.fit_report["proxy"].get(x)
        for start, snap in sorted(starters.items()):
            row = {"arm": arm, "race_id": race_id, "compound": x, "start_compound": start, "driver": snap.driver,
                   "pace_source": entry and entry["pace_source"], "wear_source": entry and entry["wear_source"]}
            try:
                t = optimize_batch(snap, true_world, seed=0, n_samples=DECISION_SAMPLES, max_stops=DECISION_STOPS)
                h = optimize_batch(snap, hidden, seed=0, n_samples=DECISION_SAMPLES, max_stops=DECISION_STOPS)
            except NoFeasiblePlan as exc:
                rows.append({**row, "error": str(exc)})
                continue
            b = true_world.simulate_batch(snap, [h.plan], seed=0, n_samples=DECISION_SAMPLES)
            r8 = not b.valid[0]
            if r8:   # the proxied plan breaks a true-world rule (R8, the race's own limit): score it without R8
                rules = [r for r in true_world.config.rules if r != "stint_limit"]
                b = true_world.simulate_batch(snap, [h.plan], seed=0, n_samples=DECISION_SAMPLES, rules=rules)
            rows.append({**row, "true_plan": json.dumps(t.plan.as_pairs()), "proxied_plan": json.dumps(h.plan.as_pairs()),
                         "plan_changed": t.plan.as_pairs() != h.plan.as_pairs(),
                         "stops_true": t.plan.num_stops, "stops_proxied": h.plan.num_stops,
                         "cost_s": float(b.times[0].mean() - t.evaluation.mean_time_s),
                         "invalid_in_true_world": r8, "true_world_errors": "; ".join(
                             true_world.simulate_batch(snap, [h.plan], seed=0, n_samples=1).errors[0]),
                         "laps_on_x_true": _laps_on(t.plan, x), "laps_on_x_proxied": _laps_on(h.plan, x),
                         "laps_on_x_diff": _laps_on(h.plan, x) - _laps_on(t.plan, x)})
    return rows


def decision_summary(dec: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm, g in dec.groupby("arm", sort=False):
        v = g[~g["invalid_in_true_world"].astype(bool)]
        out.append({"arm": arm, "rows": len(g), "plan_changed": int(g["plan_changed"].sum()),
                    "invalid_in_true_world": int(g["invalid_in_true_world"].sum()),
                    "cost_mean_s": float(v["cost_s"].mean()), "cost_median_s": float(v["cost_s"].median()),
                    "cost_max_s": float(v["cost_s"].max()), "laps_more": int((g["laps_on_x_diff"] > 0).sum()),
                    "laps_fewer": int((g["laps_on_x_diff"] < 0).sum()),
                    "laps_same": int((g["laps_on_x_diff"] == 0).sum())})
    return pd.DataFrame(out)


def r8_rows(race_id) -> list[dict]:
    """The R8 limit of each compound no car raced, beside the practice run it came from."""
    from .engine import Simulator
    data = load(race_id)
    world = Simulator(data.frame, data.config, data.practice, qualifying=data.qualifying)
    p = data.practice
    rows = []
    for c, src in sorted(world.stint_limit_sources.items()):
        on = p[p["compound"] == c] if p is not None and len(p) else p
        longest = (on.groupby(["session", "driver", "stint"]).size().max()
                   if on is not None and len(on) else np.nan)
        oldest = on.loc[on["tyre_age"].idxmax()] if on is not None and len(on) and on["tyre_age"].notna().any() else None
        rows.append({"race_id": race_id, "compound": c, "source": src, "limit_age": world.stint_limits[c],
                     "margin": data.config.stint_limit_margin_laps,
                     "cap_end_age": world.stint_limits[c] + (data.config.stint_limit_margin_laps or 0),
                     "oldest_practice_session": None if oldest is None else oldest["session"],
                     "oldest_practice_driver": None if oldest is None else oldest["driver"],
                     "longest_practice_run_laps": longest})
    return rows


def write_blend(weights, sd_floor, mse, path=STEPS_OUT):
    """Put the shipped weights, the blend's sd floor and their provenance into the table."""
    doc = json.loads(Path(path).read_text())
    doc["weights"] = weights
    doc["sd_floor"] = sd_floor
    doc["blend_provenance"] = {
        "weights": "1 / mean squared error of each source alone in the hidden-compound validation over every "
                   "calibration race (fresh-pace error at mid-race for pace, wear error for wear; qualifying "
                   "informs pace only)", "mse": mse,
        "sd_floor": "root mean squared error of the blend in the same validation, leave-one-race-out (the "
                    "weights and the step table from the other races)",
        "command": "python -m src.sim.compound_steps validate"}
    Path(path).write_text(json.dumps(doc, indent=1, default=float) + "\n")


def validate_main(argv):
    ap = argparse.ArgumentParser(prog="compound_steps validate", description="hidden-compound validation")
    ap.add_argument("--races", nargs="+", default=list(CALIBRATION_RACES))
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    ap.add_argument("--no-decisions", action="store_true", help="parameter errors only")
    ap.add_argument("--no-write", action="store_true", help="leave the weights in the table as they are")
    args = ap.parse_args(argv)
    from . import cross_race
    cross_race.guard_held_out(args.races)
    os.makedirs(args.out_dir, exist_ok=True)
    per_race = per_race_table(args.races)
    src = pd.DataFrame([r for rid in args.races for r in source_rows(rid, per_race)])
    blend = pd.DataFrame([r for rid in args.races for r in blend_rows(rid, per_race, src)])
    params = pd.concat([src, blend], ignore_index=True)
    params.to_csv(os.path.join(args.out_dir, "validation_parameters.csv"), index=False)
    table = summary(params)
    table.to_csv(os.path.join(args.out_dir, "validation_summary.csv"), index=False)
    weights, mse = weights_from(src)
    b = table.set_index("case").loc["blend"]
    sd_floor = {"pace": float(b["pace_err_s_rmse"]), "wear": float(b["wear_err_s_per_lap_rmse"])}
    loro_w = pd.DataFrame([{"race_id": rid, **{f"{k}:{s}": w for k, ws in weights_from(src, exclude=[rid])[0].items()
                                               for s, w in ws.items()}} for rid in args.races])
    loro_w.to_csv(os.path.join(args.out_dir, "weights_leave_one_race_out.csv"), index=False)
    print(params.round(3).to_string(index=False))
    print(table.round(3).to_string(index=False))
    print("weights", json.dumps(weights, indent=1), "mse", json.dumps(mse, indent=1), "sd_floor", sd_floor)
    if not args.no_write:
        write_blend(weights, sd_floor, mse)
    loro = pd.DataFrame([{"race_id": rid, **{f"{pair}:{k}": e["value"] for pair, entry in
                                             aggregate(per_race, exclude=[rid]).items() for k, e in entry.items()}}
                         for rid in args.races])
    loro.to_csv(os.path.join(args.out_dir, "steps_leave_one_race_out.csv"), index=False)
    per_race.to_csv(os.path.join(args.out_dir, "steps_by_race.csv"), index=False)
    r8 = pd.DataFrame([r for rid in args.races for r in r8_rows(rid)])
    r8.to_csv(os.path.join(args.out_dir, "r8_unraced.csv"), index=False)
    print(r8.to_string(index=False))
    if not args.no_decisions:
        dec = pd.DataFrame([r for arm in DECISION_ARMS for rid in args.races
                            for r in decision_rows(rid, per_race, src, arm)])
        dec.to_csv(os.path.join(args.out_dir, "validation_decisions.csv"), index=False)
        ds = decision_summary(dec)
        ds.to_csv(os.path.join(args.out_dir, "validation_decisions_summary.csv"), index=False)
        print(ds.round(2).to_string(index=False))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "estimate":
        return estimate_main(argv[1:])
    if argv and argv[0] == "validate":
        return validate_main(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
