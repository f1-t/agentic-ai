"""Adapters from FastF1 laps to the frame the simulator fits on.

The simulator itself never imports FastF1. Only `load_session` does, lazily.
Frame columns: driver, lap, stint, compound, tyre_age, lap_time_s, track_status,
pit_in, pit_out, start_s, end_s, team, position, pace_resid_s, driver_offset_s, clean.
practice_frame adds session, stint_len and lap_in_stint, with lap NaN.
"""
import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
LEGACY_OUTLIER_FACTOR = 1.07  # baseline rule: drop laps slower than 107% of the compound's fastest
HINGE_KNOT = 25               # tyre age where the filter's reference model may bend upwards
MIN_SCALE_S = 0.05            # floor on the robust residual scale, so noise free data cuts nothing


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


def pace_design(d, comps, knot=HINGE_KNOT, min_samples=5, lap_term=True):
    """Design matrix of the filter's reference pace model on the rows `d`.

    Per compound intercept, tyre age slope and, when at least `min_samples` of its rows
    are past the knot, a hinge `max(0, tyre_age - knot)`; then the race lap (when
    `lap_term`); then one offset per driver after the first. Returns (X, drivers) with
    the driver columns last.
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
    cols += [(d["driver"] == drv).to_numpy(dtype=float) for drv in drivers[1:]]
    return np.column_stack(cols), drivers


def robust_filter(df, base, design, k_resid=2.5, n_iter=2, min_samples=5):
    """Mark `base` laps clean unless unusually slow against a reference pace model.

    `design(rows, compounds)` builds the model's design matrix and returns (X, drivers),
    with one column per driver after the first at the end (see pace_design). A lap is
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
        X, drivers = design(d, comps)
        y = d["lap_time_s"].to_numpy()
        keep = np.ones(len(d), dtype=bool)
        for _ in range(n_iter + 1):
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
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


def practice_frame(sessions, k_resid=2.5, n_iter=2, min_samples=5, knot=HINGE_KNOT, min_stint=5):
    """Practice long-run laps from `sessions` ({"FP1": Laps, ...}) in the simulator frame.

    Base laps are green flag, non pit laps with a lap time, tyre age and compound, in
    stints with at least `min_stint` such laps (`stint_len`, counted per session, driver
    and stint). They are cleaned by robust_filter across all sessions together, with the
    race frame's reference model less the race lap term: per compound intercept, tyre
    age and hinge, and per driver offsets. `lap_in_stint` is the lap's place in its
    stint, out lap included. `lap` is NaN, as practice has no race lap (fuel) term.
    """
    keys = ["session", "driver", "stint"]
    df = pd.concat([_columns(laps).assign(session=name) for name, laps in sessions.items()],
                   ignore_index=True)
    green = _green(df)
    df["stint_len"] = green.groupby([df[k] for k in keys], dropna=False).transform("sum").astype(int)
    df["lap_in_stint"] = df.groupby(keys, dropna=False)["lap"].rank(method="first").astype(int)
    base = green & (df["stint_len"] >= min_stint)

    def design(d, comps):
        return pace_design(d, comps, knot=knot, min_samples=min_samples, lap_term=False)

    df = robust_filter(df, base, design, k_resid=k_resid, n_iter=n_iter, min_samples=min_samples)
    df["lap"] = np.nan
    extra = ["session", "stint_len", "lap_in_stint"]
    return df[[c for c in df.columns if c not in extra] + extra]


def race_stints(frame, driver):
    """Actual stints for a driver as (compound, laps, start_age, first_lap)."""
    g = frame[frame["driver"] == driver].sort_values("lap")
    out = []
    for _, s in g.groupby("stint", sort=True):
        out.append((s["compound"].iloc[0], len(s), int(s["tyre_age"].iloc[0]) - 1, int(s["lap"].iloc[0])))
    return out
