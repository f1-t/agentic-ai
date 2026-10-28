"""Pre-race pace term: per driver pace offsets from practice long runs, for the position gate.

The pooled lap time model gives every car the same pace, so a replayed race ranks drivers
by strategy alone. `calibration.positions(..., pace=offsets)` adds offset_s per lap driven
to each replayed time. Four predictors fill that term (PACE_KINDS):
  none      no offset, the pooled model
  practice  practice_pace: each driver's median residual on practice long runs
  team      team_offset: the constructor mean of the practice offsets
  insample  the race's own clean laps (calibration.run pace_offset_s): the ceiling, as it
            uses the laps being predicted

practice_pace fits nothing itself. It takes the practice fit: the degradation model of a
Simulator with fit_window "pre_race" on a data.practice_frame, fitted on the practice long
runs alone. Each long-run lap's residual is its lap time less that model at (compound, tyre
age, lap_in_stint) less the fit's session evolution term. Practice has no race lap, so the
race fuel term is not identified there: linear_fuel holds the in-stint fuel slope at the
config prior and linear has none, so whatever fuel effect the prior misses sits in the
wear slope and the base. The base delta and that absorbed fuel are common to every
driver, so they cancel in the offsets, which are relative to the field: a driver's offset
is the median residual over his long-run laps less the median of those medians over the
drivers with at least min_laps laps.
"""
import numpy as np
import pandas as pd

from .tyre_sets import stand_in
from .weather import RAIN_THRESHOLD, lap_weather

PACE_KINDS = ("none", "practice", "team", "insample")
PRACTICE_SESSIONS = ("FP2", "FP3")
WET_LAP_SHARE = 0.5   # a session is wet when more than this share of its laps are wet


def wet_session(laps, weather, rain_threshold=RAIN_THRESHOLD, share=WET_LAP_SHARE):
    """True when more than `share` of a session's laps are wet in weather.lap_weather terms
    (rain_frac above rain_threshold). `laps` is a FastF1 Laps table, `weather` the
    data.session_weather feed. Each driver's lap is its own window, as practice lap
    numbers do not line up across drivers."""
    if laps is None or len(laps) == 0:
        return False
    frame = pd.DataFrame({"lap": np.arange(len(laps)),
                          "start_s": laps["LapStartTime"].dt.total_seconds().to_numpy(),
                          "end_s": laps["Time"].dt.total_seconds().to_numpy()})
    return bool(lap_weather(weather, frame, rain_threshold)["rain"].mean() > share)


def attribute_reserves(practice, roster):
    """Practice rows with a reserve's laps relabelled as the race driver's he drove for.

    `roster` is the race drivers, driver -> team. Per session and team, the rule is
    tyre_sets.stand_in: when a race driver is the only one of his team without laps and
    exactly one other code ran for the team, that code's laps are his. Returns (frame, notes).
    """
    out, notes = practice.copy(), []
    teams = {t: set(d) for t, d in roster.groupby(roster).groups.items()}
    for session, g in practice.groupby("session"):
        for team, regulars in teams.items():
            ran = set(g.loc[g["team"] == team, "driver"].dropna())
            for driver in sorted(regulars - ran):
                code = stand_in(ran, regulars, driver)
                if code is not None:
                    out.loc[(out["session"] == session) & (out["driver"] == code), "driver"] = driver
                    notes.append(f"{session}: no {driver} laps, took {code} laps ({team})")
    return out, notes


def _residuals(d, model):
    pred = np.full(len(d), np.nan)
    for comp, idx in d.groupby("compound").indices.items():
        rows = d.iloc[idx]
        pred[idx] = model.lap_times(comp, rows["tyre_age"].to_numpy(dtype=float),
                                    rows["lap_in_stint"].to_numpy(dtype=float))
    evo = (getattr(model, "fit_report", None) or {}).get("evolution_s_per_min") or {}
    if evo:
        pred += d["session"].map(evo).fillna(0.0).to_numpy() * d["stint_start_min"].fillna(0.0).to_numpy()
    return d["lap_time_s"].to_numpy(dtype=float) - pred


def practice_pace(practice, model, min_laps=5, roster=None):
    """Per driver practice pace offsets. Returns a frame indexed by driver with columns
    offset_s, n_laps, team, source.

    `practice` is a data.practice_frame; its clean long-run laps on compounds `model`
    knows are used (a session with only one compound is fine; laps on a compound the model
    does not know are left out). `model` is the practice fit (see the module docstring).
    `roster` (race driver -> team), when given, sets the drivers and teams returned and
    lets a reserve's laps count for the driver he stood in for (attribute_reserves).
    Source is "practice" for drivers with at least `min_laps` laps; the others take their
    team-mate's offset (source "team"), or 0 with source "none" when no team-mate has one.
    """
    notes = []
    if roster is not None:
        practice, notes = attribute_reserves(practice, roster)
    use = (practice["clean"].astype(bool) & (practice["run_class"] == "long_run")
           & practice["compound"].isin(model.compounds))
    d = practice[use]
    by = pd.Series(_residuals(d, model), index=d["driver"].to_numpy()).groupby(level=0)
    n, med = by.size(), by.median()
    if roster is not None:
        drivers, team = list(roster.index), roster
    else:
        team = practice.groupby("driver")["team"].first()
        drivers = list(team.index)
    out = pd.DataFrame({"n_laps": n.reindex(drivers).fillna(0).astype(int),
                        "team": team.reindex(drivers)}, index=pd.Index(drivers, name="driver"))
    measured = out["n_laps"] >= min_laps
    field = float(med[out.index[measured]].median()) if measured.any() else 0.0
    out["offset_s"] = (med.reindex(drivers) - field).where(measured)
    out["source"] = np.where(measured, "practice", "none")
    for drv in out.index[~measured]:
        mates = out[(out["team"] == out.loc[drv, "team"]) & measured]
        if len(mates):
            out.loc[drv, ["offset_s", "source"]] = [float(mates["offset_s"].mean()), "team"]
    out["offset_s"] = out["offset_s"].fillna(0.0).astype(float)
    out = out[["offset_s", "n_laps", "team", "source"]]
    out.attrs["notes"] = notes  # reserve attributions made
    return out


def team_offset(df):
    """Constructor mean of practice_pace offsets, per driver (driver -> offset_s)."""
    return df.groupby("team")["offset_s"].transform("mean").rename("offset_s")


def load_practice(season, gp, sessions=PRACTICE_SESSIONS, long_run=None, cache_dir=None):
    """data.practice_frame of the dry `sessions` from the cache. Returns (frame or None, notes);
    a wet session (wet_session) is left out and noted."""
    from .data import CACHE_DIR, load_session, practice_frame, session_weather
    laps, notes = {}, []
    for name in sessions:
        s = load_session(season, gp, name, cache_dir=cache_dir or CACHE_DIR, weather=True)
        if wet_session(s.laps, session_weather(s)):
            notes.append(f"{name}: wet, left out")
            continue
        laps[name] = s.laps
    return (practice_frame(laps, long_run) if laps else None), notes


def practice_model(sim, practice):
    """The practice fit: sim's config with fit_window "pre_race", fitted on `practice`."""
    from .engine import Simulator
    return Simulator(sim.laps, sim.config.model_copy(update={"fit_window": "pre_race"}), practice).degradation


def offsets(sim, frame, race, practice=None, min_laps=5):
    """The four predictors' pace series (driver -> offset_s, None for "none") and the
    practice_pace table they came from. "practice" and "team" need `practice`."""
    from .calibration import run
    roster = frame.groupby("driver")["team"].first()
    paces = {"none": None, "insample": run(sim, frame, race)[0]["pace_offset_s"]}
    table = None
    if practice is not None:
        table = practice_pace(practice, practice_model(sim, practice), min_laps, roster)
        paces["practice"], paces["team"] = table["offset_s"], team_offset(table)
    return paces, table


def gate_table(sim, frame, race, results, paces):
    """Position gate under both orders for each pace series, with the grid baseline.
    Rows in mode order (simulated_order, vs_real), then pace in PACE_KINDS order, then grid."""
    from .calibration import positions
    rows, base = [], None
    for kind in [k for k in PACE_KINDS if k in paces]:
        table, summ = positions(sim, frame, race, results, pace=paces[kind])
        for mode in ("simulated_order", "vs_real"):
            rows.append({"mode": mode, "pace": kind, "n_drivers": summ["n_drivers"], **summ["by_mode"][mode]})
        err = (table["start_pos"] - table["real_pos"]).abs().dropna()
        base = {"pace": "grid", "n_drivers": int(len(err)), "within_2_rate": round(float((err <= 2).mean()), 3),
                "mae_pos": round(float(err.mean()), 2), "exact_rate": round(float((err == 0).mean()), 3)}
    rows += [{"mode": m, **base} for m in ("simulated_order", "vs_real")] if base else []
    order = {"simulated_order": 0, "vs_real": 1}
    out = pd.DataFrame(rows, columns=["mode", "pace", "n_drivers", "within_2_rate", "mae_pos", "exact_rate"])
    return out.sort_values("mode", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def offsets_table(table, insample):
    """Practice next to in-sample offsets on the race finishers (`insample`'s index), with
    in-sample also relative to the finishers' median. Returns (frame, correlation dict)."""
    t = table.reindex(insample.index)
    out = pd.DataFrame({
        "team": t["team"], "source": t["source"], "n_laps": t["n_laps"],
        "practice_s": t["offset_s"], "team_s": team_offset(table).reindex(insample.index),
        "insample_s": insample, "insample_rel_s": insample - insample.median(),
    })
    own = out[out["source"] == "practice"]
    corr = {
        "n_drivers": int(len(own)),
        "pearson": round(float(own["practice_s"].corr(own["insample_s"])), 3),
        "spearman": round(float(own["practice_s"].corr(own["insample_s"], method="spearman")), 3),
        "pearson_team": round(float(out["team_s"].corr(out["insample_s"])), 3),
    }
    return out, corr
