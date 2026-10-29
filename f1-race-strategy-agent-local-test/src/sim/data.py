"""Adapters from FastF1 laps to the frame the simulator fits on.

The simulator itself never imports FastF1. Only `load_session` does, lazily.
Frame columns: driver, lap, stint, compound, tyre_age, lap_time_s, track_status,
pit_in, pit_out, start_s, end_s, team, position, pace_resid_s, driver_offset_s, clean.
practice_frame adds session, stint_len, lap_in_stint, session_time_min, stint_start_min
and run_class, with lap NaN.
"""
import os

import numpy as np
import pandas as pd

from .config import HINGE_KNOT, LongRun   # HINGE_KNOT lives in config, shared with linear_fuel_hinge

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
LEGACY_OUTLIER_FACTOR = 1.07  # baseline rule: drop laps slower than 107% of the compound's fastest
MIN_SCALE_S = 0.05            # floor on the robust residual scale, so noise free data cuts nothing
EVOLUTION_TAU_S_PER_MIN = 0.02  # ridge toward 0 on the practice session evolution slopes


WEATHER_COLUMNS = ["time_s", "air_temp_c", "track_temp_c", "humidity_pct", "pressure_hpa",
                   "rainfall", "wind_speed_ms", "wind_dir_deg"]


def load_session(season, gp, kind="R", cache_dir=CACHE_DIR, weather=False, messages=False):
    import fastf1
    fastf1.set_log_level("ERROR")
    fastf1.Cache.configure(cache_dir=cache_dir)
    fastf1.Cache.offline_mode(True)
    session = fastf1.get_session(season, gp, kind)
    session.load(telemetry=False, weather=weather, messages=messages)
    return session


def session_weather(session):
    """Weather samples (about one per minute) in simulator units, empty if not loaded."""
    from fastf1.exceptions import DataNotLoadedError
    try:
        w = session.weather_data
    except DataNotLoadedError:  # weather=False, or absent from the cache
        w = None
    if w is None or len(w) == 0:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    return pd.DataFrame({
        "time_s": w["Time"].dt.total_seconds().to_numpy(),
        "air_temp_c": w["AirTemp"].to_numpy(),
        "track_temp_c": w["TrackTemp"].to_numpy(),
        "humidity_pct": w["Humidity"].to_numpy(),
        "pressure_hpa": w["Pressure"].to_numpy(),  # FastF1 reports mbar, same as hPa
        "rainfall": w["Rainfall"].fillna(False).astype(bool).to_numpy(),
        "wind_speed_ms": w["WindSpeed"].to_numpy(),
        "wind_dir_deg": w["WindDirection"].to_numpy(),
    })


def session_results(session):
    """Grid and classified positions by driver code, empty if the results did not load."""
    r = session.results
    if r is None or len(r) == 0:
        return pd.DataFrame(columns=["grid", "classified"])
    # ClassifiedPosition is text, with letters for retirements and disqualifications
    return pd.DataFrame({
        "grid": pd.to_numeric(r["GridPosition"], errors="coerce").to_numpy(),
        "classified": pd.to_numeric(r["ClassifiedPosition"], errors="coerce").to_numpy(),
    }, index=pd.Index(r["Abbreviation"].to_numpy(), name="driver"))


def _columns(laps):
    return pd.DataFrame({
        "driver": laps["Driver"].to_numpy(),
        "lap": laps["LapNumber"].astype(int).to_numpy(),
        "stint": laps["Stint"].to_numpy(),
        "compound": laps["Compound"].to_numpy(),
        "tyre_age": laps["TyreLife"].to_numpy(),
        "lap_time_s": laps["LapTime"].dt.total_seconds().to_numpy(),
        "track_status": laps["TrackStatus"].astype(str).to_numpy(),
        "pit_in": laps["PitInTime"].notna().to_numpy(),
        "pit_out": laps["PitOutTime"].notna().to_numpy(),
        "start_s": laps["LapStartTime"].dt.total_seconds().to_numpy(),
        "end_s": laps["Time"].dt.total_seconds().to_numpy(),
        "team": laps["Team"].to_numpy(),
        "position": laps["Position"].to_numpy(),
    })


def _green(df):
    """Green flag, non pit laps with a lap time, tyre age and compound."""
    return (
        (df["track_status"] == "1") & ~df["pit_in"] & ~df["pit_out"]
        & df["lap_time_s"].notna() & df["tyre_age"].notna() & df["compound"].notna()
    )


def evolution_columns(d):
    """Per session track evolution columns: the stint's start time (stint_start_min) on
    that session's rows, 0 elsewhere. Constant within a stint, so not collinear with tyre
    age; the slopes are identified across stints only, and fitted with a ridge toward 0
    (EVOLUTION_TAU_S_PER_MIN). Returns (sessions, columns); empty without session and
    stint_start_min."""
    if "session" not in d or "stint_start_min" not in d:
        return [], []
    t = d["stint_start_min"].to_numpy(dtype=float)
    has = ~np.isnan(t)
    sessions = sorted(d.loc[has, "session"].dropna().unique())
    return sessions, [((d["session"] == s).to_numpy() & has) * np.nan_to_num(t) for s in sessions]


def ridge_lstsq(X, y, prec=None):
    """argmin ||y - X b||^2 + sum_j prec_j b_j^2: least squares with a ridge toward 0 on
    the columns where prec > 0, plain least squares when there are none."""
    if prec is None or not np.any(prec):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    pen = np.asarray(prec, dtype=float) > 0
    rows = np.diag(np.sqrt(np.asarray(prec, dtype=float)))[pen]
    return np.linalg.lstsq(np.vstack([X, rows]), np.concatenate([y, np.zeros(pen.sum())]), rcond=None)[0]


def pace_design(d, comps, knot=HINGE_KNOT, min_samples=5, lap_term=True, session_term=False):
    """Design matrix of the filter's reference pace model on the rows `d`.

    Per compound intercept, tyre age slope and, when at least `min_samples` of its rows
    are past the knot, a hinge `max(0, tyre_age - knot)`; then the race lap (when
    `lap_term`); then a per session slope in stint start time (when `session_term`, see
    evolution_columns); then one offset per driver after the first. Returns (X, drivers,
    prec): the driver columns last, and per column ridge precisions, 1/tau^2 on the
    evolution columns and 0 elsewhere.
    """
    drivers = sorted(d["driver"].unique())
    cols = []
    age = d["tyre_age"].to_numpy(dtype=float)
    for c in comps:
        m = (d["compound"] == c).to_numpy(dtype=float)
        cols += [m, m * age]
        if (m * (age > knot)).sum() >= min_samples:
            cols.append(m * np.maximum(0.0, age - knot))
    if lap_term:
        cols.append(d["lap"].to_numpy(dtype=float))
    evo = evolution_columns(d)[1] if session_term else []
    prec = np.zeros(len(cols) + len(evo) + len(drivers) - 1)
    prec[len(cols):len(cols) + len(evo)] = EVOLUTION_TAU_S_PER_MIN ** -2
    cols += evo
    cols += [(d["driver"] == drv).to_numpy(dtype=float) for drv in drivers[1:]]
    return np.column_stack(cols), drivers, prec


def robust_filter(df, base, design, k_resid=2.5, n_iter=2, min_samples=5):
    """Mark `base` laps clean unless unusually slow against a reference pace model.

    `design(rows, compounds)` builds the model's design matrix and returns (X, drivers,
    prec), with one column per driver after the first at the end and per column ridge
    precisions (see pace_design), fitted by ridge_lstsq. A lap is
    cut when its residual is at least `k_resid` robust standard deviations (scaled MAD,
    floored at MIN_SCALE_S) above the fit; fast laps are never cut. The model is
    refitted on the clean laps and the cut redone `n_iter` times. Compounds with fewer
    than `min_samples` base laps cannot be modelled, so their base laps stay clean.
    Sets pace_resid_s, driver_offset_s and clean on `df` and returns it.
    """
    df["pace_resid_s"] = np.nan
    df["driver_offset_s"] = np.nan
    counts = df.loc[base, "compound"].value_counts()
    comps = sorted(counts.index[counts >= min_samples])
    fit = base & df["compound"].isin(comps)
    clean = base.copy()
    if fit.any():
        d = df[fit]
        X, drivers, prec = design(d, comps)
        y = d["lap_time_s"].to_numpy()
        keep = np.ones(len(d), dtype=bool)
        for _ in range(n_iter + 1):
            beta = ridge_lstsq(X[keep], y[keep], prec)
            resid = y - X @ beta
            scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), MIN_SCALE_S)
            keep = resid < k_resid * scale
        clean[fit] = keep
        df.loc[fit, "pace_resid_s"] = resid
        effects = np.concatenate([[0.0], beta[len(beta) - len(drivers) + 1:]])
        offsets = dict(zip(drivers, effects - effects.mean()))
        df["driver_offset_s"] = df["driver"].map(offsets).astype(float)
    df["clean"] = clean
    return df


def lap_frame(laps, outlier_factor=None, k_resid=2.5, n_iter=2, min_samples=5, knot=HINGE_KNOT):
    """Convert a FastF1 Laps table into the simulator frame.

    Base laps are green flag, non pit laps after lap 1 with a lap time, tyre age and
    compound. A base lap is clean unless robust_filter cuts it against a pace model:
    per compound intercept and tyre age slope, a per compound hinge `max(0, tyre_age -
    knot)` so wear may steepen on old tyres, one race lap slope (fuel and track
    evolution) and a per driver offset, fitted by least squares (pace_design). A
    compound gets no hinge when fewer than `min_samples` of its base laps are past the
    knot. The hinge lives in this reference model only, not in the simulator's
    degradation model.

    If `outlier_factor` is given the baseline rule is applied instead: a base lap is
    clean if it is faster than `outlier_factor` times its compound's fastest base lap.
    It exists for the baseline tools and test case only.
    """
    df = _columns(laps)
    base = _green(df) & (df["lap"] > 1)
    if outlier_factor is not None:
        df["pace_resid_s"] = np.nan
        df["driver_offset_s"] = np.nan
        fastest = df["lap_time_s"].where(base).groupby(df["compound"]).transform("min")
        df["clean"] = base & (df["lap_time_s"] < fastest * outlier_factor)
        return df

    def design(d, comps):
        return pace_design(d, comps, knot=knot, min_samples=min_samples)

    return robust_filter(df, base, design, k_resid=k_resid, n_iter=n_iter, min_samples=min_samples)


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, c = np.asarray(values)[order], np.cumsum(np.asarray(weights)[order])
    return float(v[np.searchsorted(c, c[-1] / 2)])


def _run_class(df, timed, sid, rule):
    """Long-run test per stint: "short", "inconsistent", "off_pace" (first failed) or "long_run"."""
    t = df[timed].assign(sid=sid[timed])
    g = t.groupby("sid")["lap_time_s"]
    st = pd.DataFrame({
        "session": t.groupby("sid")["session"].first(), "n": g.size(), "median": g.median(),
        "share": g.agg(lambda x: float((np.abs(x - x.median()) <= rule.consistent_band_s).mean())),
    })
    st["cls"] = np.where(st["n"] < rule.min_timed_laps, "short",
                         np.where(st["share"] >= rule.consistent_share, "", "inconsistent"))
    for _, ses in st.groupby("session"):
        ok = ses[ses["cls"] == ""]
        if len(ok):
            ref = _weighted_median(ok["median"].to_numpy(), ok["n"].to_numpy())
            gap = (ok["median"] - ref).abs()
            st.loc[ok.index, "cls"] = np.where(gap <= rule.max_pace_gap_s, "long_run", "off_pace")
    return sid.map(st["cls"]).fillna("short")


def practice_frame(sessions, long_run: LongRun | None = None, k_resid=2.5, n_iter=2, min_samples=5,
                   knot=HINGE_KNOT):
    """Practice long-run laps from `sessions` ({"FP1": Laps, ...}) in the simulator frame.

    Timed laps are green flag, non pit laps (so never the out or in lap) with a lap time,
    tyre age, compound and session time; `stint_len` counts them per session, driver and
    stint. A stint is a long run unless, with the thresholds in `long_run`
    (SimConfig.long_run), it fails one of these tests, checked in order:
      short         fewer than min_timed_laps timed laps
      inconsistent  under consistent_share of its timed laps are within consistent_band_s
                    of the stint median (push and cool laps alternating)
      off_pace      its median timed lap is more than max_pace_gap_s from the session's
                    long-run reference: the median of the medians of the stints that pass
                    the first two tests, weighted by their timed laps.
    `run_class` holds the result. Only long-run timed laps are base laps. They are cleaned
    by robust_filter across all sessions together, with the race frame's reference model
    less the race lap term and plus a per session slope in `stint_start_min` (see
    evolution_columns): per compound intercept, tyre age and hinge, session evolution
    between stints, and per driver offsets. `session_time_min` is minutes since the
    session's first lap started, `stint_start_min` the same for the stint's first lap.
    `lap_in_stint` is the lap's place in its stint, out lap included. `lap` is NaN, as
    practice has no race lap; the degradation fit uses lap_in_stint for the in-stint fuel.
    """
    rule = long_run or LongRun()
    keys = ["session", "driver", "stint"]
    df = pd.concat([_columns(laps).assign(session=name) for name, laps in sessions.items()],
                   ignore_index=True)
    started = df["start_s"].fillna(df["end_s"] - df["lap_time_s"])
    df["session_time_min"] = (started - started.groupby(df["session"]).transform("min")) / 60.0
    timed = _green(df) & df["session_time_min"].notna()
    sid = df.groupby(keys, dropna=False).ngroup()
    df["stint_len"] = timed.groupby(sid).transform("sum").astype(int)
    df["lap_in_stint"] = df.groupby(sid)["lap"].rank(method="first").astype(int)
    df["stint_start_min"] = df["session_time_min"].groupby(sid).transform("min")
    df["run_class"] = _run_class(df, timed, sid, rule)
    base = timed & (df["run_class"] == "long_run")

    def design(d, comps):
        return pace_design(d, comps, knot=knot, min_samples=min_samples, lap_term=False, session_term=True)

    df = robust_filter(df, base, design, k_resid=k_resid, n_iter=n_iter, min_samples=min_samples)
    df["lap"] = np.nan
    extra = ["session", "stint_len", "lap_in_stint", "session_time_min", "stint_start_min", "run_class"]
    return df[[c for c in df.columns if c not in extra] + extra]


def race_stints(frame, driver):
    """Actual stints for a driver as (compound, laps, start_age, first_lap)."""
    g = frame[frame["driver"] == driver].sort_values("lap")
    out = []
    for _, s in g.groupby("stint", sort=True):
        out.append((s["compound"].iloc[0], len(s), int(s["tyre_age"].iloc[0]) - 1, int(s["lap"].iloc[0])))
    return out
