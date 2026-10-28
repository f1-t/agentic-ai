"""Lap time models. A model maps (compound, tyre age, lap number) to a green-flag lap time.

`draw(rng)` returns a concrete lap time function for one sample. Deterministic models
return themselves, uncertainty-aware models return a parameter draw.

The fit frame may mix race laps with practice laps. Practice rows have `lap` NaN: they
count for compound base and tyre wear only, never for the race lap (fuel) term.
`fit_report` on a fitted model says what the fit saw: rows per compound, race rows,
where the fuel slope came from, and the condition number of the design matrix.
"""
import numpy as np

from .registry import Registry

DEGRADATION = Registry("degradation model")


def _usable(data, compound, min_samples):
    d = data[(data["compound"] == compound) & data["clean"]]
    return d if len(d) >= min_samples else None


def _compound_columns(d, comps):
    """Per compound indicator and indicator * tyre age, in that order."""
    cols = []
    for c in comps:
        m = (d["compound"] == c).to_numpy(dtype=float)
        cols += [m, m * d["tyre_age"].to_numpy(dtype=float)]
    return cols


def _race_lap(d):
    """Race lap numbers, NaN for practice rows (and for frames without a lap column)."""
    return d["lap"].to_numpy(dtype=float) if "lap" in d else np.full(len(d), np.nan)


def _report(d, comps, X, fuel_slope_source):
    return {
        "n_rows": {c: int((d["compound"] == c).sum()) for c in comps},
        "n_race_rows": int((~np.isnan(_race_lap(d))).sum()),
        "fuel_slope_source": fuel_slope_source,
        "condition_number": float(np.linalg.cond(X)) if X.size else float("nan"),
    }


@DEGRADATION.register("linear")
class LinearDegradation:
    """Per compound line of lap time against tyre age. Matches the baseline fit."""

    def __init__(self, coefs, fit_report=None):
        self.coefs = coefs  # compound -> (base_s, slope_s_per_lap)
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, **_):
        coefs = {}
        for comp in sorted(data["compound"].unique()):
            d = _usable(data, comp, min_samples)
            if d is not None:
                slope, base = np.polyfit(d["tyre_age"].astype(float), d["lap_time_s"], 1)
                coefs[comp] = (float(base), float(slope))
        d = data[data["compound"].isin(list(coefs)) & data["clean"]]
        X = np.column_stack(_compound_columns(d, sorted(coefs))) if coefs else np.empty((0, 0))
        return cls(coefs, _report(d, sorted(coefs), X, None))

    @property
    def compounds(self):
        return set(self.coefs)

    def draw(self, rng):
        return self

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float)


@DEGRADATION.register("linear_fuel")
class LinearFuelDegradation:
    """Linear in tyre age per compound, plus one shared slope against race lap.

    The lap term absorbs fuel burn, so tyre slopes are not confounded with the car
    getting lighter. Fitted jointly by least squares. Practice rows (lap NaN) add nothing
    to the lap column. When no compound has min_samples race rows the lap term cannot be
    identified, so the slope is fixed to `fuel_prior` (SimConfig.pre_race_fuel_slope_s_per_lap)
    and base and wear are fitted on lap times with the prior's contribution taken off.
    """

    def __init__(self, coefs, fuel_slope, fit_report=None):
        self.coefs = coefs
        self.fuel_slope = fuel_slope  # seconds per race lap, expected negative
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, fuel_prior=-0.045, **_):
        comps = [c for c in sorted(data["compound"].unique()) if _usable(data, c, min_samples) is not None]
        d = data[data["compound"].isin(comps) & data["clean"]]
        if not comps:
            return cls({}, fuel_prior, _report(d, comps, np.empty((0, 0)), "prior"))
        lap = _race_lap(d)
        race = ~np.isnan(lap)
        lap = np.where(race, lap, 0.0)
        y = d["lap_time_s"].to_numpy(dtype=float)
        n_race = [int(((d["compound"] == c).to_numpy() & race).sum()) for c in comps]
        cols = _compound_columns(d, comps)
        if any(n >= min_samples for n in n_race):
            X = np.column_stack(cols + [lap])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            fuel, source = float(beta[-1]), "fitted"
        else:
            X = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(X, y - fuel_prior * lap, rcond=None)
            fuel, source = float(fuel_prior), "prior"
        coefs = {c: (float(beta[2 * i]), float(beta[2 * i + 1])) for i, c in enumerate(comps)}
        return cls(coefs, fuel, _report(d, comps, X, source))

    @property
    def compounds(self):
        return set(self.coefs)

    def draw(self, rng):
        return self

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float) + self.fuel_slope * np.asarray(laps, dtype=float)
