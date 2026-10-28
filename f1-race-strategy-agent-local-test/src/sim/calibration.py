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
"""
import warnings

import numpy as np
import pandas as pd

from ..contracts import Plan, RaceParams, Snapshot
from .data import race_stints
from .engine import Simulator
from .events import Scenario

REPLAY_RULES = ["lap_sum", "known_compounds"]


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


def _finisher_ids(frame, min_laps_frac):
    """Drivers who completed enough laps to compare, retirements are not comparable."""
    last = frame.groupby("driver")["lap"].max()
    return sorted(last.index[last >= min_laps_frac * last.max()])


def run(sim: Simulator, frame: pd.DataFrame, race: RaceParams, min_laps_frac=0.9):
    """Returns (per driver table, summary dict, residuals by lap type)."""
    rows, laps = [], []
    for d in _finisher_ids(frame, min_laps_frac):
        g, res = replay_driver(sim, frame, race, d)
        pred = np.array(res.lap_times_s)
        n = len(g)
        real = float(g["end_s"].iloc[-1] - g["start_s"].iloc[0])
        clean = g["clean"].to_numpy()
        offset = float(np.median(g["lap_time_s"].to_numpy()[clean] - pred[clean]))
        err = res.race_time_s - real
        rows.append({
            "driver": d, "laps": n, "stops": res.num_stops, "real_s": real, "pred_s": res.race_time_s,
            "err_s": err, "err_pct": 100 * err / real, "pace_offset_s": offset,
            "err_offset_s": err + offset * n,
        })
        actual = g["lap_time_s"].to_numpy().copy()
        actual[0] = g["end_s"].iloc[0] - g["start_s"].iloc[0]  # lap 1 has no LapTime
        laps.append(g[["driver", "lap", "track_status", "pit_in", "clean"]].assign(
            kind=_lap_kind(g), resid_s=actual - pred, actual_s=actual))
    table = pd.DataFrame(rows).set_index("driver")
    laps = pd.concat(laps)
    by_kind = _by_lap_type(laps, len(table))
    summary = _summary(table)
    summary["observed_sc_factor"] = _observed_factor(laps, "sc", sim.ref_lap_s)
    summary["observed_vsc_factor"] = _observed_factor(laps, "vsc", sim.ref_lap_s)
    summary["observed_pit_loss_s"] = round(by_kind.attrs["observed_pit_loss_s"], 2)
    return table, summary, by_kind


def _observed_factor(laps, kind, ref_lap_s):
    """Median actual lap time over laps of this kind as a multiple of the reference lap, NaN if none."""
    actual = laps.loc[laps["kind"] == kind, "actual_s"].dropna()
    return round(float(actual.median() / ref_lap_s), 3) if len(actual) else float("nan")


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
        "spearman_raw": round(float(t["pred_s"].rank().corr(t["real_s"].rank())), 3),
    }


# --- team-mate and finishing position checks ------------------------------------
TIE_S = 1.0  # team-mate gaps under this are too close to call a sign


def _finishers(sim, frame, race, min_laps_frac=0.9):
    """Per finisher laps driven, real race time and replayed time (pooled model, no pace offset)."""
    rows = []
    for d in _finisher_ids(frame, min_laps_frac):
        g, res = replay_driver(sim, frame, race, d)
        rows.append({"driver": d, "team": g["team"].iloc[0], "laps": len(g),
                     "real_s": float(g["end_s"].iloc[-1] - g["start_s"].iloc[0]), "pred_s": res.race_time_s})
    return pd.DataFrame(rows).set_index("driver")


def pairwise(sim: Simulator, frame: pd.DataFrame, race: RaceParams):
    """Team-mate gap check. Returns (pairs table, summary dict).

    Team-mates share a car, so the pooled model's gap between them is mostly strategy.
    The gap is the first driver alphabetically minus the second.
    """
    fin = _finishers(sim, frame, race)
    rows, skipped = [], 0
    for team, t in fin.groupby("team"):
        if len(t) > 2:
            warnings.warn(f"{team}: {len(t)} finishers in one team, skipped")
            continue
        if len(t) < 2:
            continue  # a team-mate retired
        a, b = sorted(t.index)
        if t.loc[a, "laps"] != t.loc[b, "laps"]:
            skipped += 1
            continue
        real = t.loc[a, "real_s"] - t.loc[b, "real_s"]
        pred = t.loc[a, "pred_s"] - t.loc[b, "pred_s"]
        tie = abs(real) < TIE_S
        rows.append({"team": team, "driver_a": a, "driver_b": b, "laps": int(t.loc[a, "laps"]),
                     "real_gap_s": real, "pred_gap_s": pred, "tie": tie,
                     "sign_agree": pd.NA if tie else bool(np.sign(real) == np.sign(pred)),
                     "abs_err_s": abs(pred - real)})
    pairs = pd.DataFrame(rows, columns=[
        "team", "driver_a", "driver_b", "laps", "real_gap_s", "pred_gap_s", "tie", "sign_agree", "abs_err_s"])
    pairs["sign_agree"] = pairs["sign_agree"].astype("boolean")
    gap_err = pairs["pred_gap_s"] - pairs["real_gap_s"]
    summary = {
        "n_pairs": int(len(pairs)),
        "n_skipped_unequal_laps": skipped,
        "sign_agree_rate": round(float(pairs["sign_agree"].dropna().astype(float).mean()), 3),
        "mae_gap_s": round(float(pairs["abs_err_s"].mean()), 2),
        "bias_gap_s": round(float(gap_err.mean()), 2),
    }
    return pairs.set_index("team"), summary


def _end_positions(frame, which):
    """Lap-position of each driver on their first or last lap."""
    idx = frame.groupby("driver")["lap"].agg("idxmin" if which == "first" else "idxmax")
    return frame.loc[idx].set_index("driver")["position"]


def positions(sim: Simulator, frame: pd.DataFrame, race: RaceParams, results: pd.DataFrame | None = None):
    """Finishing order check. Returns (table, summary dict).

    pred_pos ranks each driver by laps driven, then by their replayed time against the
    others' real times. results is `data.session_results`, used when it covers every
    finisher, otherwise grid and finish come from the lap-1 and last-lap Position.
    """
    fin = _finishers(sim, frame, race)
    laps, real_s, pred_s = fin["laps"].to_numpy(), fin["real_s"].to_numpy(), fin["pred_s"].to_numpy()
    ahead = (laps[None, :] > laps[:, None]) | ((laps[None, :] == laps[:, None]) & (real_s[None, :] < pred_s[:, None]))
    np.fill_diagonal(ahead, False)  # only other finishers count
    fin["pred_pos"] = 1 + ahead.sum(axis=1)

    have = (results is not None and fin.index.isin(results.index).all()
            and results.loc[fin.index, ["grid", "classified"]].notna().all().all())
    if have:
        fin["start_pos"], fin["real_pos"] = results.loc[fin.index, "grid"], results.loc[fin.index, "classified"]
    else:
        fin["start_pos"], fin["real_pos"] = _end_positions(frame, "first"), _end_positions(frame, "last")
    fin["abs_err"] = (fin["pred_pos"] - fin["real_pos"]).abs()
    base = (fin["start_pos"] - fin["real_pos"]).abs().dropna()
    err = fin["abs_err"].dropna()
    summary = {
        "n_drivers": int(len(fin)),
        "within_2_rate": round(float((err <= 2).mean()), 3),
        "mae_pos": round(float(err.mean()), 2),
        "exact_rate": round(float((err == 0).mean()), 3),
        "baseline_within_2_rate": round(float((base <= 2).mean()), 3),
        "baseline_mae": round(float(base.mean()), 2),
        "source": "grid_classified" if have else "lap_proxies",
    }
    return fin[["team", "laps", "real_s", "pred_s", "start_pos", "real_pos", "pred_pos", "abs_err"]], summary
