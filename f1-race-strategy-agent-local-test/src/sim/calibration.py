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
import warnings

import numpy as np
import pandas as pd

from ..contracts import Plan, RaceParams, Snapshot, StintRecord
from .config import SimConfig
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
        [g["lap"] == 1, g["pit_in"], g["pit_out"], ts.str.contains("4"), ts.str.contains("[67]"),
         g["clean"], ts.str.contains("2")],
        ["lap1", "in_lap", "out_lap", "sc", "vsc", "clean", "yellow"], default="slow_green")


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
            kind=kind, resid_s=actual - pred, actual_s=actual))
    table = pd.DataFrame(rows).set_index("driver")
    laps = pd.concat(laps)
    by_kind = _by_lap_type(laps, len(table))
    summary = _summary(table)
    # the engine applies one factor to every SC lap, so the mean is the value that nets the SC total to zero
    summary["observed_sc_factor"] = _observed_factor(laps, "sc", sim.ref_lap_s, "mean")
    summary["observed_sc_factor_median"] = _observed_factor(laps, "sc", sim.ref_lap_s)
    summary["observed_vsc_factor"] = _observed_factor(laps, "vsc", sim.ref_lap_s)
    # lap 1 residual is net of the lap1_extra_s term in the replay, add it back for the absolute value
    lap1 = laps.loc[laps["kind"] == "lap1", "resid_s"]
    summary["observed_lap1_extra_s"] = (
        round(float(lap1.median()) + sim.config.effects.lap1_extra_s, 2) if len(lap1) else float("nan"))
    summary["observed_pit_loss_s"] = round(by_kind.attrs["observed_pit_loss_s"], 2)
    summary["observed_sc_pit_loss_factor"], summary["n_sc_stops"] = _sc_pit_factor(
        laps, by_kind.attrs["observed_pit_loss_s"])
    return table, summary, by_kind


def _sc_pit_factor(laps, green_pit_loss_s):
    """In plus out lap residual on stops whose in lap is under SC, over the green observed pit loss."""
    ins = laps[(laps["kind"] == "in_lap") & laps["track_status"].str.contains("4")]
    outs = laps[laps["kind"] == "out_lap"].set_index(["driver", "lap"])["resid_s"]
    cost = [r.resid_s + outs.get((r.driver, r.lap + 1), np.nan) for r in ins.itertuples()]
    cost = pd.Series(cost, dtype=float).dropna()
    if cost.empty or not green_pit_loss_s:
        return float("nan"), 0
    return round(float(cost.mean() / green_pit_loss_s), 3), int(len(cost))


def _observed_factor(laps, kind, ref_lap_s, how="median"):
    """Median (or mean) actual lap time over laps of this kind as a multiple of the reference lap, NaN if none."""
    actual = laps.loc[laps["kind"] == kind, "actual_s"].dropna()
    return round(float(getattr(actual, how)() / ref_lap_s), 3) if len(actual) else float("nan")


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


def positions(sim: Simulator, frame: pd.DataFrame, race: RaceParams, results: pd.DataFrame | None = None,
              mode: str = "simulated_order"):
    """Finishing order check. Returns (table, summary dict).

    Both orders rank by laps driven first. "simulated_order" (default) then ranks replayed
    times among themselves, so a bias common to every driver cancels. "vs_real" ranks each
    driver's replayed time against the others' real times, an absolute check that any common
    bias shifts. `mode` picks which one fills pred_pos, abs_err and the top level summary
    keys; the summary's `by_mode` holds both. results is `data.session_results`, used when it
    covers every finisher, otherwise grid and finish come from the lap-1 and last-lap Position.
    """
    if mode not in ("simulated_order", "vs_real"):
        raise ValueError(f"unknown positions mode {mode!r}")
    fin = _finishers(sim, frame, race)
    laps, real_s, pred_s = fin["laps"].to_numpy(), fin["real_s"].to_numpy(), fin["pred_s"].to_numpy()
    more = laps[None, :] > laps[:, None]
    same = laps[None, :] == laps[:, None]
    for name, ahead in (("vs_real", more | (same & (real_s[None, :] < pred_s[:, None]))),
                        ("simulated_order", more | (same & (pred_s[None, :] < pred_s[:, None])))):
        np.fill_diagonal(ahead, False)  # only other finishers count
        fin["pred_pos_" + name] = 1 + ahead.sum(axis=1)
    fin["pred_pos"] = fin["pred_pos_" + mode]

    have = (results is not None and fin.index.isin(results.index).all()
            and results.loc[fin.index, ["grid", "classified"]].notna().all().all())
    if have:
        fin["start_pos"], fin["real_pos"] = results.loc[fin.index, "grid"], results.loc[fin.index, "classified"]
    else:
        fin["start_pos"], fin["real_pos"] = _end_positions(frame, "first"), _end_positions(frame, "last")
    fin["abs_err"] = (fin["pred_pos"] - fin["real_pos"]).abs()
    base = (fin["start_pos"] - fin["real_pos"]).abs().dropna()

    def stats(col):
        err = (fin[col] - fin["real_pos"]).abs().dropna()
        return {"within_2_rate": round(float((err <= 2).mean()), 3), "mae_pos": round(float(err.mean()), 2),
                "exact_rate": round(float((err == 0).mean()), 3)}

    by_mode = {m: stats("pred_pos_" + m) for m in ("simulated_order", "vs_real")}
    summary = {
        "n_drivers": int(len(fin)),
        "mode": mode,
        **by_mode[mode],
        "baseline_within_2_rate": round(float((base <= 2).mean()), 3),
        "baseline_mae": round(float(base.mean()), 2),
        "source": "grid_classified" if have else "lap_proxies",
        "by_mode": by_mode,
    }
    return fin[["team", "laps", "real_s", "pred_s", "start_pos", "real_pos", "pred_pos",
                "pred_pos_vs_real", "pred_pos_simulated_order", "abs_err"]], summary


# --- mid-race replay ------------------------------------------------------------
# At lap k the simulator is fitted only on what its fit window allows by then and
# replays each finisher's real remaining strategy. The plan is the real one, so the
# error measures the physics as a function of horizon and fit window. run() is the
# k = 0, full window corner.
MIDRACE_WINDOWS = ("full", "to_snapshot", "pre_race")
MIN_OFFSET_LAPS = 5      # green non-pit laps up to k needed for a driver pace offset


def sim_version(config: SimConfig) -> str:
    """Physics config hash: the first 12 hex digits of sha256 over the SimConfig as JSON with
    sorted keys, `fit_window` left out because each report row names its window.

    Identical configs give the same value on any machine or commit; a change to any config
    field (degradation, effects, shrinkage, long_run, ...) changes it. Code changes that keep
    the config do not, so it identifies the physics settings, not the code version.
    """
    import hashlib
    import json
    data = config.model_dump(mode="json", exclude={"fit_window"})
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
BUCKET_LAPS = 10         # horizon bucket width in aggregate()
N_BOOT = 500             # race-cluster bootstrap draws


def _current_stint(stints, k):
    """Index into race_stints of the stint lap k + 1 is driven in, and how many of its laps are done.

    A stint whose last lap is k is finished: the driver pitted at the end of lap k.
    """
    for j, (_, n, _, first) in enumerate(stints):
        if k + 1 < first + n:
            return j, max(0, k + 1 - first)
    raise ValueError(f"no stint covers lap {k + 1}")


def _set_age(g, k, start_age, done):
    """Laps on the current set after lap k: TyreLife of the lap-k row while still on that set."""
    if done == 0:
        return start_age
    row = g[g["lap"] == k]
    if len(row) and pd.notna(row["tyre_age"].iloc[0]):
        return int(row["tyre_age"].iloc[0])
    return start_age + done


def _track_status(ts):
    return "sc" if "4" in ts else "vsc" if ("6" in ts or "7" in ts) else "green"


def snapshot_at(frame: pd.DataFrame, race: RaceParams, driver: str, k: int) -> Snapshot:
    """The driver's state after lap k, read off the lap frame.

    Stand-in for the data team's snapshot builder, kept minimal. Race length is the laps
    the driver drove, as in replay_driver. The caller skips drivers without a lap-k row.
    """
    g = frame[frame["driver"] == driver].sort_values("lap")
    stints = race_stints(frame, driver)
    j, done = _current_stint(stints, k)
    comp, _, start_age, _ = stints[j]
    position, status, elapsed = None, "green", 0.0
    if k > 0:
        row = g[g["lap"] == k].iloc[0]
        position = int(row["position"]) if pd.notna(row["position"]) else None
        status = _track_status(str(row["track_status"]))
        elapsed = float(row["end_s"] - g["start_s"].iloc[0])
    hist = _scenario(g[g["lap"] <= k])
    return Snapshot(
        race=race.model_copy(update={"total_laps": len(g)}), driver=driver, laps_completed=k,
        compound=comp, tyre_age=_set_age(g, k, start_age, done),
        stints=[StintRecord(compound=c, laps=n, start_lap=first, start_age=a) for c, n, a, first in stints[:j]],
        sc_laps=hist.sc_laps, vsc_laps=hist.vsc_laps, track_status=status, elapsed_s=elapsed,
        position=position, sets_available=None, standing_start=k == 0)


def remaining_plan(frame: pd.DataFrame, driver: str, k: int) -> Plan:
    """The driver's real strategy from lap k + 1: the rest of the current set, then the later stints."""
    g = frame[frame["driver"] == driver]
    stints = race_stints(frame, driver)
    j, done = _current_stint(stints, k)
    comp, n, start_age, _ = stints[j]
    first = [comp, n - done, _set_age(g, k, start_age, done)]
    return Plan.of([first] + [[c, m, a] for c, m, a, _ in stints[j + 1:]])


def default_laps_k(frame: pd.DataFrame, total_laps: int) -> list[int]:
    """Every 5th lap from 0 to total_laps - 10, plus the lap before and after each SC and VSC period."""
    laps = set(range(0, total_laps - 10 + 1, 5))
    status = frame["track_status"].astype(str)
    for pattern in ("4", "[67]"):  # SC, VSC
        hit = sorted(int(x) for x in frame.loc[status.str.contains(pattern), "lap"].unique())
        for i, lap in enumerate(hit):
            if i == 0 or hit[i - 1] != lap - 1:
                laps.add(lap - 1)
            if i == len(hit) - 1 or hit[i + 1] != lap + 1:
                laps.add(lap + 1)
    return sorted(x for x in laps if 0 <= x < total_laps)


def _pace_offset(sim: Simulator, g: pd.DataFrame, k: int) -> float:
    """Median actual minus predicted lap time over the driver's green non-pit laps up to k.

    Uses only what is known at lap k: green flag, no pit in or out, after lap 1, under the
    window's own model. Not the frame's clean flag, whose filter saw the whole race. The
    median shrugs off the few traffic laps that this lets in. NaN below MIN_OFFSET_LAPS.
    """
    c = g[(g["lap"] <= k) & (g["lap"] > 1) & (g["track_status"].astype(str) == "1") & ~g["pit_in"]
          & ~g["pit_out"] & g["lap_time_s"].notna() & g["tyre_age"].notna() & g["compound"].isin(sim.modelled)]
    if len(c) < MIN_OFFSET_LAPS:
        return float("nan")
    pred = np.empty(len(c))
    for comp, idx in c.groupby("compound").indices.items():
        sub = c.iloc[idx]
        pred[idx] = sim.degradation.lap_times(comp, sub["tyre_age"].to_numpy(dtype=float),
                                              sub["lap"].to_numpy(dtype=float))
    return float(np.median(c["lap_time_s"].to_numpy(dtype=float) - pred))


MIDRACE_COLUMNS = ["race_id", "window", "driver", "team", "k", "laps", "horizon", "elapsed_s",
                   "real_remaining_s", "pred_remaining_s", "err_s", "pace_offset_s", "err_offset_s",
                   "pred_pos_sim", "pred_pos_real", "real_pos", "abs_pos_err_sim", "abs_pos_err_real"]


def _real_positions(frame, finishers, results):
    """Classified position when `results` covers every finisher, as positions() does, else last-lap Position."""
    if (results is not None and pd.Index(finishers).isin(results.index).all()
            and results.loc[finishers, "classified"].notna().all()):
        return results.loc[finishers, "classified"].astype(float)
    return _end_positions(frame, "last").reindex(finishers).astype(float)


def _rank(laps, totals, against):
    """1 + finishers ahead: more laps driven, or the same laps and a lower `against` total. NaN totals rank NaN."""
    laps, totals, against = (np.asarray(x, dtype=float) for x in (laps, totals, against))
    ahead = (laps[None, :] > laps[:, None]) | ((laps[None, :] == laps[:, None]) & (against[None, :] < totals[:, None]))
    np.fill_diagonal(ahead, False)  # only other finishers count
    return np.where(np.isnan(totals), np.nan, 1 + ahead.sum(axis=1))


def midrace(sim_factory, frame: pd.DataFrame, race: RaceParams, laps_k=None, windows=MIDRACE_WINDOWS,
            practice=None, min_laps_frac=0.9, results: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (window, finisher, k): predicted against real remaining race time.

    sim_factory(window) returns the Simulator for this race with that fit window. None
    means Simulator(frame, SimConfig(fit_window=window), practice). laps_k defaults to
    default_laps_k. A plan the window cannot model (unknown compound) gets NaN predictions.

    Positions rank laps driven first, as positions() does. pred_pos_sim ranks elapsed plus
    predicted remaining among the other finishers' predicted totals at the same k, so a
    bias common to every driver cancels; pred_pos_real ranks it against their real totals.
    real_pos is the classified position from `results` (data.session_results) when it
    covers every finisher, otherwise the last-lap Position.
    """
    if sim_factory is None:
        def sim_factory(window):
            return Simulator(frame, SimConfig(fit_window=window), practice)
    if laps_k is None:
        laps_k = default_laps_k(frame, race.total_laps)
    groups = {d: frame[frame["driver"] == d].sort_values("lap") for d in _finisher_ids(frame, min_laps_frac)}
    fin = pd.DataFrame({
        "laps": [len(g) for g in groups.values()],
        "real_s": [float(g["end_s"].iloc[-1] - g["start_s"].iloc[0]) for g in groups.values()],
        "team": [g["team"].iloc[0] for g in groups.values()],
    }, index=list(groups))
    fin["real_pos"] = _real_positions(frame, list(groups), results)
    rows = []
    for window in windows:
        sim = sim_factory(window)
        for k in laps_k:  # k outer so a to_snapshot fit is built once per lap
            at_k = []
            for d, g in groups.items():
                if k >= len(g) or (k > 0 and not (g["lap"] == k).any()):
                    continue
                snap = snapshot_at(frame, race, d, k)
                s = sim.fitted_for(snap)
                res = s.simulate(snap, remaining_plan(frame, d, k), scenario=_scenario(g[g["lap"] > k]),
                                 rules=REPLAY_RULES, return_laps=True)
                real_rem = fin.loc[d, "real_s"] - snap.elapsed_s
                pred_rem = res.remaining.mean_s if res.valid else float("nan")
                err, offset, horizon = pred_rem - real_rem, _pace_offset(s, g, k), snap.remaining_laps
                at_k.append({
                    "race_id": race.race_id, "window": window, "driver": d, "team": fin.loc[d, "team"],
                    "k": k, "laps": len(g), "horizon": horizon, "elapsed_s": snap.elapsed_s,
                    "real_remaining_s": real_rem, "pred_remaining_s": pred_rem, "err_s": err,
                    "pace_offset_s": offset, "err_offset_s": err + offset * horizon,
                    "real_pos": fin.loc[d, "real_pos"]})
            if not at_k:
                continue
            t = pd.DataFrame(at_k)
            pred_total = (t["elapsed_s"] + t["pred_remaining_s"]).to_numpy()
            # rows missing a lap-k row are not simulated at k, so the simulated order is among those that are
            t["pred_pos_sim"] = _rank(t["laps"], pred_total, np.where(np.isnan(pred_total), np.inf, pred_total))
            # against every other finisher's real total
            pos_real = []
            for d, laps, total in zip(t["driver"], t["laps"], pred_total):
                others = fin.drop(index=d)
                ahead = (others["laps"] > laps) | ((others["laps"] == laps) & (others["real_s"] < total))
                pos_real.append(np.nan if np.isnan(total) else 1 + int(ahead.sum()))
            t["pred_pos_real"] = pos_real
            t["abs_pos_err_sim"] = (t["pred_pos_sim"] - t["real_pos"]).abs()
            t["abs_pos_err_real"] = (t["pred_pos_real"] - t["real_pos"]).abs()
            rows.append(t)
    if not rows:
        return pd.DataFrame(columns=MIDRACE_COLUMNS)
    return pd.concat(rows, ignore_index=True)[MIDRACE_COLUMNS]


def _bucketed(rows):
    return rows.assign(horizon_bucket=(rows["horizon"] // BUCKET_LAPS * BUCKET_LAPS).astype(int))


def _metrics(t):
    err = t["err_s"].dropna()
    out = {
        "n": int(len(err)),
        "n_nan": int(t["pred_remaining_s"].isna().sum()),
        "n_races": int(t["race_id"].nunique()),
        "mae_s": float(err.abs().mean()),
        "bias_s": float(err.mean()),
        "mae_offset_s": float(t["err_offset_s"].abs().mean()),
    }
    for mode in ("sim", "real"):
        pos = t[f"abs_pos_err_{mode}"].dropna()
        out[f"within_2_rate_{mode}"] = float((pos <= 2).mean())
        out[f"mae_pos_{mode}"] = float(pos.mean())
    return out


def _bootstrap_ci(t, n_boot=N_BOOT, seed=0):
    """95 percent CI for mae_offset_s, resampling whole races with replacement. NaN with one race."""
    races = t["race_id"].unique()
    if len(races) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_race = {r: t.loc[t["race_id"] == r, "err_offset_s"].abs().dropna().to_numpy() for r in races}
    stats = []
    for _ in range(n_boot):
        draw = np.concatenate([by_race[r] for r in rng.choice(races, size=len(races), replace=True)])
        if len(draw):
            stats.append(draw.mean())
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


SUMMARY_COLUMNS = ["n", "n_nan", "n_races", "mae_s", "bias_s", "mae_offset_s", "mae_offset_ci_lo",
                   "mae_offset_ci_hi", "within_2_rate_sim", "mae_pos_sim", "within_2_rate_real", "mae_pos_real"]


def _table(rows, keys, ci, n_boot=N_BOOT, seed=0):
    out, index = [], []
    for key, t in _bucketed(rows).groupby(keys, sort=True):
        m = _metrics(t)
        m["mae_offset_ci_lo"], m["mae_offset_ci_hi"] = _bootstrap_ci(t, n_boot, seed) if ci else (np.nan, np.nan)
        out.append(m)
        index.append(key)
    cols = SUMMARY_COLUMNS if ci else [c for c in SUMMARY_COLUMNS if not c.startswith("mae_offset_ci")]
    table = pd.DataFrame(out, columns=cols, index=pd.MultiIndex.from_tuples(index, names=keys))
    return table.astype({"n": int, "n_nan": int, "n_races": int})


def aggregate(rows: pd.DataFrame, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """Summary indexed by (window, horizon_bucket); bucket b holds horizons b to b + 9.

    mae_offset_ci_lo and _hi are a race-cluster bootstrap 95 percent CI, NaN with one race.
    """
    return _table(rows, ["window", "horizon_bucket"], ci=True, n_boot=n_boot, seed=seed)


def per_race(rows: pd.DataFrame) -> pd.DataFrame:
    """aggregate()'s metrics per (race_id, window, horizon_bucket), for the report appendix."""
    return _table(rows, ["race_id", "window", "horizon_bucket"], ci=False)


def by_driver(rows: pd.DataFrame) -> pd.DataFrame:
    """aggregate()'s metrics per (race_id, window, driver, horizon_bucket), for the report appendix."""
    return _table(rows, ["race_id", "window", "driver", "horizon_bucket"], ci=False)


def midrace_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """Team-mate sign agreement, pred_remaining against real_remaining, per (window, horizon_bucket).

    pairwise()'s rules: a team counts at a k when exactly two of its finishers have rows
    there with equal laps driven; real gaps under TIE_S are ties and leave the rate out.
    The gap is the first driver alphabetically minus the second.
    """
    pairs = []
    for (_, window, _, _), t in rows.groupby(["race_id", "window", "k", "team"], sort=True):
        if len(t) > 2:
            warnings.warn(f"{t['team'].iloc[0]}: {len(t)} finishers in one team, skipped")
            continue
        if len(t) < 2 or t["laps"].nunique() != 1 or t["pred_remaining_s"].isna().any():
            continue
        a, b = t.sort_values("driver").itertuples()
        real = a.real_remaining_s - b.real_remaining_s
        pred = a.pred_remaining_s - b.pred_remaining_s
        tie = bool(abs(real) < TIE_S)
        pairs.append({"window": window, "horizon": a.horizon, "tie": tie,
                      "sign_agree": np.nan if tie else float(np.sign(real) == np.sign(pred))})
    keys = ["window", "horizon_bucket"]
    if not pairs:
        return pd.DataFrame(columns=["n_pairs", "n_ties", "sign_agree_rate"],
                            index=pd.MultiIndex.from_tuples([], names=keys))
    return _bucketed(pd.DataFrame(pairs)).groupby(keys).agg(
        n_pairs=("tie", "size"), n_ties=("tie", "sum"), sign_agree_rate=("sign_agree", "mean"))
