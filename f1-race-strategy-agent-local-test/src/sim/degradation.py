"""Lap time models. A model maps (compound, tyre age, lap number) to a green-flag lap time.

`draw(rng)` returns a concrete lap time function for one sample. Deterministic models
return themselves, uncertainty-aware models return a parameter draw.

Rows with `lap` NaN are practice laps: they count for compound base and tyre wear only,
never for the race lap (fuel) term.

`fit(..., prior=model, shrinkage=taus)` is the to_snapshot fit: race rows only, by
penalised least squares toward `prior` (the pre-race fit from practice), minimising
||y - X b||^2 + sum_j ((b_j - b0_j) / tau_j)^2, closed form b = (X'X + D)^-1 (X'y + D b0)
with D = diag(1 / tau^2). `shrinkage` holds the prior standard deviations: base_s,
wear_s_per_lap and fuel_s_per_lap. A compound the prior does not model is shrunk toward
the mean of the prior's compound coefficients with tau doubled. The fuel slope is
shrunk toward `fuel_prior`, as practice cannot identify it. With `prior` None (no
practice) only the fuel slope is penalised; base and wear are plain least squares.

`fit_report` on a fitted model says what the fit saw:
  n_rows, max_tyre_age        rows and oldest tyre per compound, practice rows behind a
                              prior included: the support of each compound's line
  n_race_rows                 race rows in the fit, total
  n_race_rows_per_compound    the same per compound
  fuel_slope_source           "fitted", "prior" (fixed, no race rows) or "shrunk"; None
                              for models without a fuel term
  condition_number            of the design matrix (race rows alone when shrunk), with
                              all-zero columns left out
  shrinkage_weight            per parameter, the data's share of the posterior precision
                              (X'X)_jj / ((X'X)_jj + D_jj): {"base_s": {compound: w},
                              "wear_s_per_lap": {compound: w}, "fuel_s_per_lap": w}.
                              1 for an unpenalised fitted parameter, 0 for a fixed prior.
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


def _cond(X):
    X = X[:, np.any(X != 0, axis=0)] if X.size else X
    return float(np.linalg.cond(X)) if X.size else float("nan")


def _report(d, comps, X, fuel_slope_source, weight, has_fuel, support=None):
    """`weight` is per design column: base and wear per compound, then fuel when has_fuel.
    `support` is the prior's fit_report, whose rows count toward n_rows and max_tyre_age."""
    race = ~np.isnan(_race_lap(d))
    support = support or {"n_rows": {}, "max_tyre_age": {}}
    rep = {"n_rows": {}, "n_race_rows": int(race.sum()), "n_race_rows_per_compound": {},
           "max_tyre_age": {}, "fuel_slope_source": fuel_slope_source, "condition_number": _cond(X),
           "shrinkage_weight": {"base_s": {}, "wear_s_per_lap": {},
                                "fuel_s_per_lap": float(weight[-1]) if has_fuel else None}}
    for i, c in enumerate(comps):
        m = (d["compound"] == c).to_numpy()
        ages = [float(d.loc[m, "tyre_age"].max())] if m.any() else []
        ages += [support["max_tyre_age"][c]] if c in support["max_tyre_age"] else []
        rep["n_rows"][c] = int(m.sum()) + support["n_rows"].get(c, 0)
        rep["n_race_rows_per_compound"][c] = int((m & race).sum())
        rep["max_tyre_age"][c] = max(ages) if ages else float("nan")
        rep["shrinkage_weight"]["base_s"][c] = float(weight[2 * i])
        rep["shrinkage_weight"]["wear_s_per_lap"][c] = float(weight[2 * i + 1])
    return rep


def _shrunk(data, min_samples, prior, shrinkage, fuel_prior=None):
    """Penalised fit of the race rows toward `prior`. Returns (comps, beta, report).

    Modelled: every compound of the prior, plus race compounds with min_samples clean
    race rows. The fuel column is added when fuel_prior is not None.
    """
    race = data[data["clean"].to_numpy(dtype=bool) & ~np.isnan(_race_lap(data))]
    pc = dict(prior.coefs) if prior is not None else {}
    counts = race["compound"].value_counts()
    comps = sorted(set(pc) | {c for c, n in counts.items() if n >= min_samples})
    d = race[race["compound"].isin(comps)]
    pooled = tuple(np.mean(list(pc.values()), axis=0)) if pc else None
    beta0, prec = [], []
    for c in comps:
        if pc:
            (b, w), k = (pc[c], 1.0) if c in pc else (pooled, 2.0)
            beta0 += [b, w]
            prec += [(k * shrinkage["base_s"]) ** -2, (k * shrinkage["wear_s_per_lap"]) ** -2]
        else:  # no practice: base and wear unpenalised
            beta0 += [0.0, 0.0]
            prec += [0.0, 0.0]
    cols = _compound_columns(d, comps)
    if fuel_prior is not None:
        cols.append(_race_lap(d))
        beta0.append(fuel_prior)
        prec.append(shrinkage["fuel_s_per_lap"] ** -2)
    X = np.column_stack(cols) if cols else np.empty((len(d), 0))
    beta0, prec = np.asarray(beta0, dtype=float), np.asarray(prec, dtype=float)
    xtx = X.T @ X
    if cols:
        rhs = X.T @ d["lap_time_s"].to_numpy(dtype=float) + prec * beta0
        beta = np.linalg.lstsq(xtx + np.diag(prec), rhs, rcond=None)[0]
    else:
        beta = beta0
    dj = np.diag(xtx)
    weight = np.divide(dj, dj + prec, out=np.ones_like(dj), where=prec > 0)
    source = None if fuel_prior is None else ("shrunk" if len(d) else "prior")
    support = prior.fit_report if prior is not None else None
    return comps, beta, _report(d, comps, X, source, weight, fuel_prior is not None, support)


def _coefs(comps, beta):
    return {c: (float(beta[2 * i]), float(beta[2 * i + 1])) for i, c in enumerate(comps)}


@DEGRADATION.register("linear")
class LinearDegradation:
    """Per compound line of lap time against tyre age. Matches the baseline fit."""

    def __init__(self, coefs, fit_report=None):
        self.coefs = coefs  # compound -> (base_s, slope_s_per_lap)
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, prior=None, shrinkage=None, **_):
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage)
            return cls(_coefs(comps, beta), rep)
        coefs = {}
        for comp in sorted(data["compound"].unique()):
            d = _usable(data, comp, min_samples)
            if d is not None:
                slope, base = np.polyfit(d["tyre_age"].astype(float), d["lap_time_s"], 1)
                coefs[comp] = (float(base), float(slope))
        comps = sorted(coefs)
        d = data[data["compound"].isin(comps) & data["clean"]]
        X = np.column_stack(_compound_columns(d, comps)) if comps else np.empty((0, 0))
        return cls(coefs, _report(d, comps, X, None, np.ones(2 * len(comps)), False))

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
    With `shrinkage`, the penalised fit toward `prior` described in the module docstring.
    """

    def __init__(self, coefs, fuel_slope, fit_report=None):
        self.coefs = coefs
        self.fuel_slope = fuel_slope  # seconds per race lap, expected negative
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, fuel_prior=-0.045, prior=None, shrinkage=None, **_):
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage, fuel_prior)
            return cls(_coefs(comps, beta), float(beta[-1]), rep)
        comps = [c for c in sorted(data["compound"].unique()) if _usable(data, c, min_samples) is not None]
        d = data[data["compound"].isin(comps) & data["clean"]]
        if not comps:
            return cls({}, fuel_prior, _report(d, comps, np.empty((0, 0)), "prior", np.zeros(1), True))
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
        weight = np.append(np.ones(2 * len(comps)), 1.0 if source == "fitted" else 0.0)
        return cls(_coefs(comps, beta), fuel, _report(d, comps, X, source, weight, True))

    @property
    def compounds(self):
        return set(self.coefs)

    def draw(self, rng):
        return self

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float) + self.fuel_slope * np.asarray(laps, dtype=float)
