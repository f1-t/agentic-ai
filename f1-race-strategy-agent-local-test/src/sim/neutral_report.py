"""Neutralisations in the world, off against on: the accuracy report behind the world default
(docs/eval.md "Pit loss and track status").

    python -m src.sim.neutral_report --races 2021_04_spain 2023_07_spain ... [--out reports/calibration/neutralised]

Per calibration race (explicit ids only) and world arm, replaying every finisher's real
strategy on the world (full window, whole-race pit loss):

  G          from each snapshot k (calibration.default_laps_k) the real remaining plan, predicted
             less real remaining time; err_offset_s nets the driver's pace offset at k as
             calibration.midrace does. Summarised as mae and signed bias of err_offset_s by
             10-lap horizon bucket.
  sc stops   every finisher's real stop whose in-lap was under SC or VSC: predicted in- plus
             out-lap time plus the stop's pit cost, less the real in- plus out-lap time.
  neutral    every finisher's non-pit lap under SC or VSC: predicted less real lap time.

Arms: off (no neutralised laps: the old world), on_factor (the recorded laps, the provisional
sc/vsc lap and pit-loss factors), on_recorded (recorded laps with the field's recorded lap times,
provisional pit-loss factors), on_recorded_fitted (as on_recorded, with the SC and VSC pit-loss
factors pooled from the other calibration races' stops in configs/sim/pit_loss_prior.json:
leave-this-race-out, so the number is out of sample). Never reads a held-out race.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .calibration import (BUCKET_LAPS, REPLAY_RULES, _finisher_ids, _pace_offset, default_laps_k, remaining_plan,
                          snapshot_at)
from .config import SimConfig
from .engine import Simulator
from .events import Scenario

ARMS = ("off", "on_factor", "on_recorded", "on_recorded_fitted")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration", "neutralised")


def _kind(status) -> str:
    s = str(status)
    return "sc" if "4" in s else "vsc" if ("6" in s or "7" in s) else "green"


def arm_config(arm: str, race_id: str, base: SimConfig | None = None) -> SimConfig:
    """The world config of `arm` for `race_id` (module doc)."""
    from . import pit_loss as _pl
    cfg = base or SimConfig()
    if arm == "off":
        return cfg.model_copy(update={"events": cfg.events.model_copy(update={"name": "none", "params": {}})})
    lap_times = "factor" if arm == "on_factor" else "recorded"
    cfg = cfg.model_copy(update={"events": cfg.events.model_copy(update={"name": "recorded",
                                                                          "params": {"lap_times": lap_times}})})
    if arm == "on_recorded_fitted":
        sc = pooled_factor("sc", cfg.pit_loss, exclude=race_id)
        vsc = pooled_factor("vsc", cfg.pit_loss, exclude=race_id)
        cfg = cfg.model_copy(update={"effects": cfg.effects.model_copy(
            update={"sc_pit_loss_factor": sc, "vsc_pit_loss_factor": vsc})})
    return cfg


def pooled_factor(kind: str, pit_loss_cfg, exclude: str | None = None) -> float:
    """Mean factor of every `kind` stop in the prior table (less race `exclude`), clipped to
    [0, 1]; the PitLossModel default without any."""
    races = (pit_loss_cfg.prior or {}).get("races", {})
    x = [f for r, e in races.items() if r != exclude for f in (e.get(kind) or {}).get("factors", [])]
    default = pit_loss_cfg.default_sc_factor if kind == "sc" else pit_loss_cfg.default_vsc_factor
    return float(np.clip(np.mean(x), 0.0, 1.0)) if x else float(default)


def _world_scenario(sim, snap, arm, g, k):
    if arm == "off":
        return Scenario()
    return sim.events.draw(snap, None)


def race_rows(data, arm: str, world_pit_loss: str = "replay_mean"):
    """(G rows, sc-stop rows, neutral-lap rows) of one race under one arm. The replays price a
    green stop at the race's replay mean (G's value) or, with world_pit_loss "median", at the
    world's (pit_loss.whole_race)."""
    cfg = arm_config(arm, data.race_id)
    sim = Simulator(data.frame, cfg, data.practice, qualifying=data.qualifying, pit_loss_s=data.race.pit_loss_s)
    if world_pit_loss == "median":
        from . import pit_loss
        sim.race_pit_loss_s = pit_loss.whole_race(data.frame, sim.degradation, data.race_id, cfg.pit_loss)["pit_loss_s"]
    f, race = data.frame, data.race
    groups = {d: f[f["driver"] == d].sort_values("lap") for d in _finisher_ids(f, 0.9)}
    g_rows, stop_rows, lap_rows = [], [], []
    for k in default_laps_k(f, race.total_laps):
        for d, g in groups.items():
            if k >= len(g) or (k > 0 and not (g["lap"] == k).any()):
                continue
            snap = snapshot_at(f, race, d, k)
            res = sim.simulate(snap, remaining_plan(f, d, k), scenario=_world_scenario(sim, snap, arm, g, k),
                               rules=REPLAY_RULES, return_laps=True)
            if not res.valid:
                continue
            real_rem = float(g["end_s"].iloc[-1] - g["start_s"].iloc[0]) - snap.elapsed_s
            offset = _pace_offset(sim, g, k)
            err = res.remaining.mean_s - real_rem
            g_rows.append({"race_id": data.race_id, "arm": arm, "driver": d, "k": k, "horizon": snap.remaining_laps,
                           "err_s": err, "pace_offset_s": offset, "err_offset_s": err + offset * snap.remaining_laps})
            if k != 0:
                continue
            pred = np.asarray(res.lap_times_s, dtype=float)
            actual = g["lap_time_s"].to_numpy(dtype=float).copy()
            actual[0] = g["end_s"].iloc[0] - g["start_s"].iloc[0]
            kinds = g["track_status"].map(_kind).to_numpy()
            pit_in, pit_out = g["pit_in"].astype(bool).to_numpy(), g["pit_out"].astype(bool).to_numpy()
            pl = sim.pit_loss_inputs(snap)
            for i in range(len(g)):
                if kinds[i] != "green" and not pit_in[i] and not pit_out[i] and np.isfinite(actual[i]):
                    lap_rows.append({"race_id": data.race_id, "arm": arm, "driver": d, "lap": int(g["lap"].iloc[i]),
                                     "kind": kinds[i], "err_s": pred[i] - actual[i]})
                if pit_in[i] and i + 1 < len(g) and pit_out[i + 1] and kinds[i] != "green":
                    if not (np.isfinite(actual[i]) and np.isfinite(actual[i + 1])):
                        continue
                    on = arm != "off"
                    factor = (pl["sc_factor"] if kinds[i] == "sc" else pl["vsc_factor"]) if on else 1.0
                    cost = factor * pl["pit_loss_s"]
                    stop_rows.append({"race_id": data.race_id, "arm": arm, "stop": f"{d}:{int(g['lap'].iloc[i])}",
                                      "kind": kinds[i], "pred_s": pred[i] + pred[i + 1] + cost,
                                      "real_s": actual[i] + actual[i + 1],
                                      "err_s": pred[i] + pred[i + 1] + cost - actual[i] - actual[i + 1]})
    return pd.DataFrame(g_rows), pd.DataFrame(stop_rows), pd.DataFrame(lap_rows)


def summarise(g, stops, laps) -> dict[str, pd.DataFrame]:
    """The three tables: G by race, arm and horizon bucket; sc stops and neutral laps by race and arm."""
    g = g.assign(bucket=(g["horizon"] // BUCKET_LAPS) * BUCKET_LAPS)

    def mb(x, col="err_s"):
        v = x[col].dropna()
        return pd.Series({"n": len(v), "mae_s": v.abs().mean(), "bias_s": v.mean()})
    out = {"g_by_horizon": g.groupby(["race_id", "arm", "bucket"]).apply(mb, "err_offset_s", include_groups=False).reset_index(),
           "g_by_race": g.groupby(["race_id", "arm"]).apply(mb, "err_offset_s", include_groups=False).reset_index(),
           "g_raw_by_race": g.groupby(["race_id", "arm"]).apply(mb, "err_s", include_groups=False).reset_index()}
    if len(stops):
        out["sc_stops"] = stops.groupby(["race_id", "arm", "kind"]).apply(mb, include_groups=False).reset_index()
    if len(laps):
        out["neutral_laps"] = laps.groupby(["race_id", "arm", "kind"]).apply(mb, include_groups=False).reset_index()
    return out


def main(argv=None):
    from .compute import load_profile
    from .jobs import load_race
    ap = argparse.ArgumentParser(prog="python -m src.sim.neutral_report")
    ap.add_argument("--races", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--world-pit-loss", choices=("replay_mean", "median"), default="replay_mean",
                    help="the green pit loss the replays price stops at (default G's replay mean)")
    args = ap.parse_args(argv)
    gs, ss, ls = [], [], []
    for rid in args.races:
        data = load_race(rid, load_profile())
        for arm in args.arms:
            g, s, lp = race_rows(data, arm, args.world_pit_loss)
            gs.append(g), ss.append(s), ls.append(lp)
    tables = summarise(pd.concat(gs), pd.concat(ss), pd.concat(ls))
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)
    for name, t in tables.items():
        t.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(f"== {name}\n{t.round(2).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
