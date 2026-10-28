"""Lap time models. A model maps (compound, tyre age, lap number) to a green-flag lap time.

`draw(rng)` returns a concrete lap time function for one sample. Deterministic models
return themselves, uncertainty-aware models return a parameter draw.
"""
import numpy as np

from .registry import Registry

DEGRADATION = Registry("degradation model")


def _usable(data, compound, min_samples):
    d = data[(data["compound"] == compound) & data["clean"]]
    return d if len(d) >= min_samples else None


@DEGRADATION.register("linear")
class LinearDegradation:
    """Per compound line of lap time against tyre age. Matches the baseline fit."""

    def __init__(self, coefs):
        self.coefs = coefs  # compound -> (base_s, slope_s_per_lap)

    @classmethod
    def fit(cls, data, min_samples=5, **_):
        coefs = {}
        for comp in sorted(data["compound"].unique()):
            d = _usable(data, comp, min_samples)
            if d is not None:
                slope, base = np.polyfit(d["tyre_age"].astype(float), d["lap_time_s"], 1)
                coefs[comp] = (float(base), float(slope))
        return cls(coefs)

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
    getting lighter. Fitted jointly by least squares.
    """

    def __init__(self, coefs, fuel_slope):
        self.coefs = coefs
        self.fuel_slope = fuel_slope  # seconds per race lap, expected negative

    @classmethod
    def fit(cls, data, min_samples=5, **_):
        comps = [c for c in sorted(data["compound"].unique()) if _usable(data, c, min_samples) is not None]
        d = data[data["compound"].isin(comps) & data["clean"]]
        cols = []
        for c in comps:
            m = (d["compound"] == c).to_numpy(dtype=float)
            cols += [m, m * d["tyre_age"].to_numpy(dtype=float)]
        cols.append(d["lap"].to_numpy(dtype=float))
        beta, *_ = np.linalg.lstsq(np.column_stack(cols), d["lap_time_s"].to_numpy(), rcond=None)
        coefs = {c: (float(beta[2 * i]), float(beta[2 * i + 1])) for i, c in enumerate(comps)}
        return cls(coefs, float(beta[-1]))

    @property
    def compounds(self):
        return set(self.coefs)

    def draw(self, rng):
        return self

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float) + self.fuel_slope * np.asarray(laps, dtype=float)
