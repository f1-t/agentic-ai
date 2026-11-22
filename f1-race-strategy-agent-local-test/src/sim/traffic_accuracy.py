"""Traffic off against on, judged by the physical accuracy measures (docs/eval.md "Traffic").

    python -m src.sim.traffic_accuracy --races 2021_04_spain 2023_07_spain ... --out reports/calibration/traffic

Per race and arm, on the real strategies and the real SC/VSC laps:

    G            calibration.midrace: the real remaining plan replayed from each snapshot lap k,
                 predicted less real remaining time. err_offset_s adds the driver's pace offset
                 at k (calibration._pace_offset, laps up to k), the repo's G; err_s is raw. mae
                 and signed bias per horizon bucket and over horizons G_HORIZONS.
    rejoin       calibration.rejoin over every real stop of a finisher: our position and gap to
                 the car ahead after the out-lap (horizon 2, REJOIN_HORIZON), and the same three
                 laps later (horizon 5), predicted against the lap frame's.
    pack         calibration.run's full-race replay: observed_pack_penalty_s, the mean residual
                 of clean laps behind a car (clean_pack) less that of clean free-air laps, under
                 each arm (what the arm leaves unexplained), and for a traffic arm the model's
                 own estimate, the mean traffic increment on clean_pack laps less that on
                 clean_free laps.
    stop_gain    calibration.stop_gain at the real stops from each k, against the arm's own
                 full-window fit: mae and signed bias (it moves only with the fit, not with
                 traffic).

World arms run the full window (the scorer, which may read the whole race); belief arms run
configs/sim/belief.yaml's to_snapshot window with the projected field (as-of only). The arms
are ARMS. Nothing here changes a lap time; it only reports.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, jobs, traffic
from .compute import load_profile
from .config import ComponentSpec, SimConfig
from .engine import Simulator

ROOT = Path(__file__).resolve().parents[2]
BELIEF = ROOT / "configs" / "sim" / "belief.yaml"
WORLD_TRAFFIC = ROOT / "configs" / "sim" / "world.yaml"
G_HORIZONS = (10, 49)
LONG_REJOIN = 5   # the rejoin three laps after the out-lap
BELIEF_SAMPLES = 8   # belief arms replay this many samples per row (draws and projection variants)

_GHOST = {"name": "ghost", "params": {"align_neutral": True}}
_OWN = {"name": "ghost", "params": {"align_neutral": True, "own_pace": True}}
_EARNED = {"name": "ghost", "params": {"align_neutral": True, "own_pace": True, "pass_mode": "earned",
                                       "dirty_air_auto": True}}
_PROJ_OWN = {"name": "ghost", "params": {"source": "projected", "own_pace": True}}
_PROJ_EARNED = {"name": "ghost", "params": {"source": "projected", "own_pace": True, "pass_mode": "earned",
                                            "dirty_air_auto": True}}


def _no_dirty(spec):
    return {**spec, "params": {**spec["params"], "dirty_air_auto": False}}


def _world(**update):
    return SimConfig.model_validate({**SimConfig().model_dump(mode="json"), **update})


def _belief(**update):
    base = SimConfig.from_file(BELIEF).model_dump(mode="json")
    return SimConfig.model_validate({**base, **update})


# arm -> (role, config factory); "world" arms run the full window, "belief" arms to_snapshot
ARMS = {
    "world_off": ("world", lambda: _world()),
    "world_ghost": ("world", lambda: _world(traffic=_GHOST)),
    "world_ghost_own": ("world", lambda: _world(traffic=_OWN)),
    "world_earned": ("world", lambda: _world(traffic=_EARNED)),
    "world_earned_free": ("world", lambda: _world(traffic=_EARNED, fit_free_air_only=True)),
    "belief_off": ("belief", lambda: _belief()),
    "belief_projected_own": ("belief", lambda: _belief(traffic=_PROJ_OWN)),
    "belief_projected_earned": ("belief", lambda: _belief(traffic=_PROJ_EARNED)),
    "world_earned_nodirty": ("world", lambda: _world(traffic=_no_dirty(_EARNED))),
    "belief_projected_earned_nodirty": ("belief", lambda: _belief(traffic=_no_dirty(_PROJ_EARNED))),
}


def _sim(data, cfg):
    return Simulator(data.frame, cfg, data.practice, qualifying=getattr(data, "qualifying", None))


def long_rejoin(sim, frame, race, horizon=LONG_REJOIN, min_laps_frac=0.9) -> pd.DataFrame:
    """calibration.rejoin with the replay `horizon` laps from the in-lap: position and gap
    to the car ahead at the end of lap in_lap + horizon - 1."""
    rows = []
    for d in calibration._finisher_ids(frame, min_laps_frac):
        g = frame[frame["driver"] == d].sort_values("lap").set_index("lap")
        for in_lap in g.index[g["pit_in"].astype(bool)]:
            k, end = int(in_lap) - 1, int(in_lap) + horizon - 1
            if end not in g.index or (k > 0 and k not in g.index):
                continue
            try:
                _, res, t = calibration._replay_window(sim, frame, race, d, k, end)
            except ValueError:
                continue
            pred_pos, pred_gap = traffic.place(frame, d, end, t)
            _, real_gap = traffic.place(frame, d, end, float(g.loc[end, "end_s"]))
            rows.append({"driver": d, "in_lap": int(in_lap), "pred_pos": pred_pos,
                         "real_pos": float(g.loc[end, "position"]), "pred_gap_ahead_s": pred_gap,
                         "real_gap_ahead_s": real_gap})
    out = pd.DataFrame(rows, columns=["driver", "in_lap", "pred_pos", "real_pos", "pred_gap_ahead_s",
                                      "real_gap_ahead_s"])
    return out.assign(pos_err=out["pred_pos"] - out["real_pos"])


def _rejoin_stats(rows, prefix):
    err = rows["pos_err"].dropna()
    gap = (rows["pred_gap_ahead_s"] - rows["real_gap_ahead_s"]).dropna()
    nan = float("nan")
    return {f"{prefix}_n": int(len(err)),
            f"{prefix}_pos_mae": float(err.abs().mean()) if len(err) else nan,
            f"{prefix}_pos_bias": float(err.mean()) if len(err) else nan,
            f"{prefix}_within_1": float((err.abs() <= 1).mean()) if len(err) else nan,
            f"{prefix}_gap_mae_s": float(gap.abs().mean()) if len(gap) else nan,
            f"{prefix}_gap_bias_s": float(gap.mean()) if len(gap) else nan}


def _pack(sim, off, data):
    """observed_pack_penalty_s under `sim`, and for a traffic arm the model's own pack
    estimate: mean (on less off) lap increment on clean_pack laps less that on clean_free."""
    _, summary, _ = calibration.run(sim, data.frame, data.race)
    out = {"pack_observed_s": summary["observed_pack_penalty_s"]}
    if sim.config.traffic.name == "none" or off is None:
        return out
    inc = {"clean_pack": [], "clean_free": []}
    for d in calibration._finisher_ids(data.frame, 0.9):
        g, on_r = calibration.replay_driver(sim, data.frame, data.race, d)
        _, off_r = calibration.replay_driver(off, data.frame, data.race, d)
        kind = calibration._lap_kind(g)
        diff = np.array(on_r.lap_times_s) - np.array(off_r.lap_times_s)
        for kk in inc:
            inc[kk].append(diff[kind == kk])
    pack, free = (np.concatenate(inc[kk]) if inc[kk] else np.empty(0) for kk in ("clean_pack", "clean_free"))
    out["pack_model_s"] = (float(pack.mean() - free.mean()) if len(pack) and len(free) else float("nan"))
    return out


def _g_stats(rows, horizons=G_HORIZONS):
    t = rows[(rows["horizon"] >= horizons[0]) & (rows["horizon"] <= horizons[1])]
    e, eo = t["err_s"].dropna(), t["err_offset_s"].dropna()
    return {"g_n": int(len(eo)), "g_mae_offset_s": float(eo.abs().mean()), "g_bias_offset_s": float(eo.mean()),
            "g_mae_raw_s": float(e.abs().mean()), "g_bias_raw_s": float(e.mean())}


def race_arm(race_id: str, arm: str, profile=None) -> tuple[dict, pd.DataFrame]:
    """(summary row, G rows) of one race under one arm."""
    profile = profile or load_profile()
    data = jobs.load_race(race_id, profile)
    role, make = ARMS[arm]
    cfg = make()
    window = "full" if role == "world" else "to_snapshot"
    cfg = cfg.model_copy(update={"fit_window": window})
    laps_k = jobs.laps_k_for(data.frame, data.race.total_laps, profile)
    n = 1 if role == "world" else BELIEF_SAMPLES
    rows = calibration.midrace(lambda w: _sim(data, cfg), data.frame, data.race, laps_k=laps_k, windows=[window],
                               results=data.results, n_samples=n)
    rows = rows.assign(race_id=race_id, arm=arm)
    sim = _sim(data, cfg)
    out = {"race_id": race_id, "arm": arm, "role": role, "config_hash": calibration.config_hash(cfg),
           **_g_stats(rows)}
    out.update(_rejoin_stats(calibration.rejoin(sim, data.frame, data.race), "rejoin2"))
    out.update(_rejoin_stats(long_rejoin(sim, data.frame, data.race), "rejoin5"))
    hind = _sim(data, cfg.model_copy(update={"fit_window": "full"})).degradation
    sg = calibration.stop_gain(sim, data.frame, data.race, hind, laps_k)
    out.update(sg_n=int(len(sg)), sg_mae_s=float(sg["err_s"].abs().mean()) if len(sg) else np.nan,
               sg_bias_s=float(sg["err_s"].mean()) if len(sg) else np.nan)
    tp = cfg.traffic.params
    if tp.get("dirty_air_auto"):   # the dirty air the world reads off the whole race
        out["dirty_air_s"] = traffic.dirty_air_estimate(data.frame, sim.degradation,
                                                        tp.get("dirty_air_range_s", traffic.FREE_AIR_GAP_S), 0.0)
    if role == "world":
        off = (_sim(data, cfg.model_copy(update={"traffic": ComponentSpec(name="none")}))
               if cfg.traffic.name != "none" else None)
        out.update(_pack(sim, off, data))
    return out, rows


def _job(item):
    race_id, arm = item
    return race_arm(race_id, arm)


def run(races, arms=tuple(ARMS), n_jobs=8):
    items = [(r, a) for r in races for a in arms]
    with ProcessPoolExecutor(n_jobs) as pool:
        results = list(pool.map(_job, items))
    summary = pd.DataFrame([r for r, _ in results])
    g = common_offset(pd.concat([rows for _, rows in results], ignore_index=True))
    common = pd.DataFrame([{"race_id": r, "arm": a, **_g_stats(t.assign(err_offset_s=t["err_common_s"]))}
                           for (r, a), t in g.groupby(["race_id", "arm"])])
    common = common.rename(columns={"g_mae_offset_s": "g_mae_common_s", "g_bias_offset_s": "g_bias_common_s"})
    summary = summary.merge(common[["race_id", "arm", "g_mae_common_s", "g_bias_common_s"]], on=["race_id", "arm"],
                            how="left")
    return summary, g


OFF_ARM = {"world": "world_off", "belief": "belief_off"}


def common_offset(g: pd.DataFrame) -> pd.DataFrame:
    """err_common_s: err_s plus the traffic-off arm's pace offset (same role, race, driver, k)
    times the horizon, so every arm is pace-corrected by one definition (a traffic arm's own
    err_offset_s uses free-air offsets, calibration.free_air_offset). NaN without that arm."""
    g = g.copy()
    g["role"] = g["arm"].map(lambda a: ARMS[a][0])
    base = g[g["arm"].isin(OFF_ARM.values())][["role", "race_id", "driver", "k", "pace_offset_s"]]
    g = g.merge(base.rename(columns={"pace_offset_s": "off_pace_offset_s"}), on=["role", "race_id", "driver", "k"],
                how="left")
    g["err_common_s"] = g["err_s"] + g["off_pace_offset_s"] * g["horizon"]
    return g


def by_horizon(g: pd.DataFrame) -> pd.DataFrame:
    """mae and signed bias of err_offset_s and err_s per (arm, horizon bucket), race-pooled."""
    g = g.assign(horizon_bucket=(g["horizon"] // calibration.BUCKET_LAPS * calibration.BUCKET_LAPS).astype(int))
    return g.groupby(["arm", "horizon_bucket"]).agg(
        n=("err_offset_s", "count"), mae_offset_s=("err_offset_s", lambda x: x.abs().mean()),
        bias_offset_s=("err_offset_s", "mean"), mae_common_s=("err_common_s", lambda x: x.abs().mean()),
        bias_common_s=("err_common_s", "mean"), mae_raw_s=("err_s", lambda x: x.abs().mean()),
        bias_raw_s=("err_s", "mean")).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.traffic_accuracy", description=__doc__.splitlines()[0])
    ap.add_argument("--races", nargs="+", required=True, help="explicit race ids")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    from .cross_race import guard_held_out
    guard_held_out(args.races)
    summary, g = run(args.races, args.arms, args.jobs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    g.to_parquet(out / "g_rows.parquet", index=False)
    by_horizon(g).to_csv(out / "g_by_horizon.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
