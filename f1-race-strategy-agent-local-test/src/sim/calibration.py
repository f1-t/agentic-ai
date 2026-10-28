"""Calibration check: replay each driver's actual strategy and compare to the real time.

The replay uses the driver's real stints (compound, length, tyre age at fitting) and
the safety car laps they actually drove under. Planning rules are switched off because
real strategies do not always respect them.

Two error views are reported per driver:
- raw: predicted minus real race time. Includes car and driver pace differences, which
  a pooled lap time model does not capture.
- offset adjusted: the same after removing a per driver pace offset taken from clean
  laps. This isolates the strategy dependent part (tyres, pit loss, neutralised laps),
  which is what matters when comparing plans for one driver.
- green: the offset adjusted error summed over green non pit laps only, leaving out
  lap 1, neutralised laps and in and out laps. Safety car and pit loss errors do not
  reach it, so it is the number that judges a lap filter.
"""
import numpy as np
import pandas as pd

from ..contracts import Plan, RaceParams, Snapshot
from .data import race_stints
from .engine import Simulator
from .events import Scenario

REPLAY_RULES = ["lap_sum", "known_compounds"]
NOT_GREEN = ["lap1", "sc", "vsc", "in_lap", "out_lap"]


def _scenario(g):
    sc = g.loc[g["track_status"].str.contains("4"), "lap"].tolist()
    vsc = g.loc[g["track_status"].str.contains("[67]"), "lap"].tolist()
    return Scenario(sc_laps=sc, vsc_laps=vsc)


def replay_driver(sim: Simulator, frame: pd.DataFrame, race: RaceParams, driver: str):
    g = frame[frame["driver"] == driver].sort_values("lap")
    stints = race_stints(frame, driver)
    comp0, _, age0, _ = stints[0]
    snap = Snapshot.at_start(race.model_copy(update={"total_laps": len(g)}), driver, comp0, tyre_age=age0)
    plan = Plan.of([[c, k, a] for c, k, a, _ in stints])
    res = sim.simulate(snap, plan, scenario=_scenario(g), rules=REPLAY_RULES, return_laps=True)
    if not res.valid:
        raise ValueError(f"{driver}: replay plan rejected: {res.errors}")
    return g, res


def _lap_kind(g):
    ts = g["track_status"]
    return np.select(
        [g["lap"] == 1, ts.str.contains("4"), ts.str.contains("[67]"), g["pit_in"], g["pit_out"],
         g["clean"], ts.str.contains("2")],
        ["lap1", "sc", "vsc", "in_lap", "out_lap", "clean", "yellow"], default="slow_green")


def run(sim: Simulator, frame: pd.DataFrame, race: RaceParams, min_laps_frac=0.9):
    """Returns (per driver table, summary dict, residuals by lap type)."""
    full = frame.groupby("driver")["lap"].max().max()
    rows, laps = [], []
    for d in sorted(frame["driver"].unique()):
        if frame.loc[frame["driver"] == d, "lap"].max() < min_laps_frac * full:
            continue  # retirements are not comparable
        g, res = replay_driver(sim, frame, race, d)
        pred = np.array(res.lap_times_s)
        n = len(g)
        real = float(g["end_s"].iloc[-1] - g["start_s"].iloc[0])
        clean = g["clean"].to_numpy()
        offset = float(np.median(g["lap_time_s"].to_numpy()[clean] - pred[clean]))
        err = res.race_time_s - real
        actual = g["lap_time_s"].to_numpy().copy()
        actual[0] = g["end_s"].iloc[0] - g["start_s"].iloc[0]  # lap 1 has no LapTime
        kind = _lap_kind(g)
        green = ~np.isin(kind, NOT_GREEN)
        rows.append({
            "driver": d, "laps": n, "stops": res.num_stops, "real_s": real, "pred_s": res.race_time_s,
            "err_s": err, "err_pct": 100 * err / real, "pace_offset_s": offset,
            "err_offset_s": err + offset * n,
            "err_green_s": float(np.nansum(pred[green] - actual[green] + offset)),
        })
        laps.append(g[["driver", "lap", "track_status", "pit_in", "clean"]].assign(
            kind=kind, resid_s=actual - pred))
    table = pd.DataFrame(rows).set_index("driver")
    return table, _summary(table), _by_lap_type(pd.concat(laps), len(table))


def _by_lap_type(df, n_drivers):
    summ = df.groupby("kind")["resid_s"].agg(["count", "mean", "median", "std"]).round(3)
    # contribution to the average driver's race total error (actual minus predicted)
    summ["total_per_driver_s"] = (df.groupby("kind")["resid_s"].sum() / n_drivers).round(1)
    # extra time on green in and out laps over the green prediction is the observed pit loss
    green_pits = df[df["kind"].isin(["in_lap", "out_lap"]) & (df["track_status"].isin(["1", "21", "12"]))]
    summ.attrs["observed_pit_loss_s"] = float(
        green_pits["resid_s"].sum() / max(green_pits["kind"].eq("in_lap").sum(), 1))
    return summ


def _summary(t):
    err = t["err_s"]
    return {
        "n_drivers": int(len(t)),
        "mae_s": round(float(err.abs().mean()), 2),
        "bias_s": round(float(err.mean()), 2),
        "rmse_s": round(float(np.sqrt((err ** 2).mean())), 2),
        "mae_pct": round(float(t["err_pct"].abs().mean()), 3),
        "mae_offset_adjusted_s": round(float(t["err_offset_s"].abs().mean()), 2),
        "bias_offset_adjusted_s": round(float(t["err_offset_s"].mean()), 2),
        "mae_green_s": round(float(t["err_green_s"].abs().mean()), 2),
        "bias_green_s": round(float(t["err_green_s"].mean()), 2),
        "spearman_raw": round(float(t["pred_s"].rank().corr(t["real_s"].rank())), 3),
    }
