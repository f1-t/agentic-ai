"""Whole-field replay: every classified car's real strategy through the simulator together.

    python -m src.sim.field_replay --races 2021_04_spain ... --traffic off on --out reports/calibration/field_replay

The acceptance measure for the integrated simulator: run every classified car's real
remaining strategy (real compounds, stint lengths and set ages, the real SC/VSC laps) from a
calibration point k (0 = the start, then laps 10, 20, 30, 40) to the flag, and compare with
the race as it was run:

    times   simulated less real total race time per car (elapsed real time to k plus the
            simulated remaining time): mae and signed bias per race and k
    order   the simulated finishing order (laps driven first, then simulated total) against
            the classification: Spearman rank correlation, mean absolute position error,
            share of cars within one position

With traffic on, the cars run together: pass 1 simulates every car without traffic, then
each further pass (n_iter) replays every car against the other cars' trajectories from the
previous pass, as ghosts (traffic.py, the recorded source on a frame whose laps after k are
the simulated ones). Laps up to k are the real ones in every pass. Cars that are not
classified keep their real laps (they retire as they did).

Fit and pace, and which numbers are in-sample:

    fit     lodo (default): leave-one-driver-out, the full-window fit without car D's laps
            (SimConfig.exclude_snapshot_driver), so D's own laps never fit D's prediction;
            insample: every car's laps fit every prediction (flatters the measure)
    pace    the driver term in the engine (Simulator.pace): practice (this weekend's
            practice long runs, pace.practice_pace: known before the race, out of sample),
            asof (D's race pace on green laps up to k, calibration._pace_offset; practice at
            k = 0 and below MIN_OFFSET_LAPS laps: out of sample for the laps it predicts),
            insample (D's whole-race offset, calibration.run: in-sample), none

Every row carries fit, pace and in_sample (True when either the fit or the pace read D's
laps after k).
"""
from __future__ import annotations

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, jobs, pace, traffic
from .compute import load_profile
from .config import ComponentSpec, SimConfig
from .engine import Simulator

KS = (0, 10, 20, 30, 40)
N_ITER = 2
PACE_KINDS = ("practice", "asof", "insample", "none")
FITS = ("lodo", "insample")
ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "configs" / "sim" / "world.yaml"


def classified(data) -> list[str]:
    """Finishers (calibration._finisher_ids) with a classified position in the results, else
    every finisher."""
    fin = calibration._finisher_ids(data.frame, 0.9)
    res = data.results
    if res is not None and "classified" in res:
        ok = [d for d in fin if d in res.index and pd.notna(res.loc[d, "classified"])]
        if ok:
            return ok
    return fin


def _sim_frame(frame, k, trajectories, pit_loss):
    """`frame` with every car in `trajectories` (driver -> (lap_times after k, plan)) re-timed
    after lap k: lap times from the simulation, the pit loss added to each in-lap's end (the
    engine books it there), pit_in / pit_out and compound from the plan. Other rows are kept."""
    parts = [frame[~frame["driver"].isin(list(trajectories))]]
    for d, (laps_s, plan) in trajectories.items():
        g = frame[frame["driver"] == d].sort_values("lap")
        keep = g[g["lap"] <= k]
        base = float(keep["end_s"].iloc[-1]) if len(keep) else float(g["start_s"].iloc[0])
        n = len(laps_s)
        lap_no = np.arange(k + 1, k + 1 + n)
        comp, pit_in, pit_out, ages = [], np.zeros(n, bool), np.zeros(n, bool), []
        i = 0
        for j, s in enumerate(plan.stints):
            a0 = s.start_age if j else None
            for m in range(s.laps):
                comp.append(s.compound)
                ages.append((a0 or 0) + m + 1)
                i += 1
            if j < len(plan.stints) - 1:
                pit_in[i - 1] = True
                if i < n:
                    pit_out[i] = True
        t = np.asarray(laps_s, dtype=float) + np.where(pit_in, pit_loss, 0.0)
        end = base + np.cumsum(t)
        start = np.concatenate([[base], end[:-1]])
        real = g.set_index("lap")
        status = [str(real["track_status"].get(L, "1")) if L in real.index else "1" for L in lap_no]
        new = pd.DataFrame({"driver": d, "team": g["team"].iloc[0], "lap": lap_no, "lap_time_s": t,
                            "start_s": start, "end_s": end, "pit_in": pit_in, "pit_out": pit_out,
                            "compound": comp, "tyre_age": ages, "track_status": status, "clean": False})
        parts += [keep, new]
    return pd.concat(parts, ignore_index=True)


def _pace_series(kind, data, off_sim):
    if kind in ("none",):
        return None
    paces, _ = pace.offsets(off_sim, data.frame, data.race, data.practice)
    if kind == "insample":
        return paces["insample"]
    return paces.get("practice")   # practice, and asof's fallback


def replay(data, cfg: SimConfig, k: int, *, fit="lodo", pace_kind="practice", n_iter=N_ITER,
           drivers=None, paces=None) -> pd.DataFrame:
    """One row per classified car at calibration point k. `cfg` is the world config (its
    traffic decides whether the cars run together); `paces` is {kind: series} to reuse."""
    frame, race = data.frame, data.race
    drivers = drivers or classified(data)
    traffic_on = cfg.traffic.name != "none"
    tp = dict(cfg.traffic.params)
    if traffic_on:   # one dirty-air value from the real race; the engine pace replaces own_pace
        if tp.pop("dirty_air_auto", False):
            model = Simulator(frame, cfg.model_copy(update={"traffic": ComponentSpec(name="none")})).degradation
            tp["dirty_air_s"] = traffic.dirty_air_estimate(frame, model, tp.get("dirty_air_range_s", 1.5), 0.0)
        tp["own_pace"] = False
        tp["source"] = "recorded"
    base = cfg.model_copy(update={"fit_window": "full", "exclude_snapshot_driver": fit == "lodo",
                                  "traffic": ComponentSpec(name="none")})
    sim = Simulator(frame, base, data.practice, qualifying=getattr(data, "qualifying", None))
    series = (paces or {}).get(pace_kind) if paces else _pace_series(pace_kind, data, sim)
    if pace_kind == "asof":
        series = series.copy() if series is not None else pd.Series(dtype=float)
    groups = {d: frame[frame["driver"] == d].sort_values("lap") for d in drivers}
    snaps, plans, scen, sims = {}, {}, {}, {}
    for d, g in groups.items():
        if k >= len(g) or (k > 0 and not (g["lap"] == k).any()):
            continue
        snaps[d] = calibration.snapshot_at(frame, race, d, k)
        plans[d] = calibration.remaining_plan(frame, d, k)
        scen[d] = calibration._scenario(g[g["lap"] > k])
        s = sim
        if pace_kind == "asof" and k > 0:
            off = calibration._pace_offset(sim.fitted_for(snaps[d]), g, k)
            if np.isfinite(off):
                series = series.copy()
                series[d] = off
        if series is not None:
            s = sim.with_pace(series)
        sims[d] = s.fitted_for(snaps[d])

    def run(field_frame, use_traffic):
        out = {}
        for d in snaps:
            s = sims[d]
            if use_traffic:
                s = copy.copy(s)
                s.laps = field_frame
                tr_cfg = s.config.model_copy(update={"traffic": ComponentSpec(name="ghost", params=tp)})
                s.config = tr_cfg
                s.traffic = traffic.GhostTraffic(**tp)
            b = s.simulate_batch(snaps[d], [plans[d]], scenario=scen[d], rules=calibration.REPLAY_RULES,
                                 n_samples=1, return_laps=True)
            if not b.valid[0]:
                raise ValueError(f"{d} at k={k}: replay rejected: {b.errors[0]}")
            out[d] = (b.lap_times[0], float(b.times[0, 0]), float(b.pit_loss[0, 0]))
        return out

    res = run(frame, False)
    for _ in range(n_iter if traffic_on else 0):
        traj = {d: (r[0], plans[d]) for d, r in res.items()}
        stops = {d: max(1, plans[d].num_stops) for d in res}
        pit = {d: res[d][2] / stops[d] for d in res}
        mean_pit = float(np.mean(list(pit.values()))) if pit else float(race.pit_loss_s)
        res = run(_sim_frame(frame, k, traj, mean_pit), True)
    rows = []
    for d, (_, rem, _) in res.items():
        g = groups[d]
        real = float(g["end_s"].iloc[-1] - g["start_s"].iloc[0])
        pred = snaps[d].elapsed_s + rem
        rows.append({"race_id": race.race_id, "k": k, "driver": d, "laps": len(g), "real_s": real, "pred_s": pred,
                     "err_s": pred - real})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["real_pos"] = calibration._real_positions(frame, list(t["driver"]), data.results).to_numpy()
    t["pred_pos"] = calibration._rank(t["laps"], t["pred_s"], t["pred_s"])
    # real_pos is among every finisher; re-rank it among the cars replayed here
    t["real_pos"] = calibration._rank(t["laps"], t["real_pos"], t["real_pos"])
    return t.assign(traffic="on" if traffic_on else "off", fit=fit, pace=pace_kind,
                    in_sample=fit == "insample" or pace_kind == "insample")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (race_id, traffic, fit, pace, k): n, time mae and bias, Spearman, position mae,
    share within one position."""
    out = []
    keys = ["race_id", "traffic", "fit", "pace", "in_sample", "k"]
    for key, t in rows.groupby(keys, sort=True):
        err = (t["pred_pos"] - t["real_pos"]).abs()
        rho = t["pred_pos"].corr(t["real_pos"], method="spearman") if len(t) >= 2 else np.nan
        out.append({**dict(zip(keys, key)), "n": len(t), "time_mae_s": float(t["err_s"].abs().mean()),
                    "time_bias_s": float(t["err_s"].mean()), "spearman": float(rho),
                    "pos_mae": float(err.mean()), "within_1": float((err <= 1).mean())})
    return pd.DataFrame(out)


def race_job(item):
    race_id, config_path, fit, pace_kinds, ks, n_iter = item
    profile = load_profile()
    data = jobs.load_race(race_id, profile)
    cfg = SimConfig.from_file(config_path) if config_path else SimConfig()
    off = Simulator(data.frame, cfg.model_copy(update={"traffic": ComponentSpec(name="none")}), data.practice)
    all_paces, _ = pace.offsets(off, data.frame, data.race, data.practice)
    paces = {"practice": all_paces.get("practice"), "asof": all_paces.get("practice"),
             "insample": all_paces["insample"], "none": None}
    parts = [replay(data, cfg, k, fit=fit, pace_kind=pk, n_iter=n_iter, paces=paces)
             for pk in pace_kinds for k in ks]
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.field_replay", description=__doc__.splitlines()[0])
    ap.add_argument("--races", nargs="+", required=True, help="explicit race ids")
    ap.add_argument("--configs", nargs="+", default=["none", str(WORLD)],
                    help="world configs to replay under ('none': SimConfig(), traffic off); default off and "
                         "configs/sim/world.yaml")
    ap.add_argument("--fit", choices=FITS, default="lodo")
    ap.add_argument("--pace", nargs="+", choices=PACE_KINDS, default=["practice", "asof"])
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    from .cross_race import guard_held_out
    guard_held_out(args.races)
    items = [(r, None if c == "none" else c, args.fit, tuple(args.pace), tuple(args.ks), args.n_iter)
             for r in args.races for c in args.configs]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = pd.concat(list(pool.map(race_job, items)), ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "rows.csv", index=False)
    summ = summarize(rows)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
