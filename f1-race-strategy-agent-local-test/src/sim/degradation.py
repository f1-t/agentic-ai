"""Lap time models. A model maps (compound, tyre age, lap number) to a green-flag lap time.

`draw(rng)` returns a concrete lap time function for one sample. Deterministic models
return themselves, uncertainty-aware models return a parameter draw.

Rows with `lap` NaN are practice laps: they count for compound base and tyre wear only,
never for the race lap (fuel) term. Inside a practice run tyre age and fuel burn advance
together, so practice alone cannot tell wear from the fuel gain: linear_fuel gives
practice rows a fuel column, lap_in_stint, with its coefficient fixed to `fuel_prior`
(the prior's part is taken off their lap times before fitting).

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
  evolution_s_per_min         per session track evolution slope, {session: s/min}, when the
                              rows carry session and stint_start_min (practice_frame);
                              {} otherwise. The simulator does not use it.
  unmodelled                  compounds in the data that are not modelled, with the reason:
                              "no long run" for practice with no long-run stint on it,
                              else the clean lap count against min_samples
  base_delta_s                {compound: s/lap} added to practice-derived bases (`base_delta`,
                              SimConfig.pre_race_base_delta_s); carried by a shrunk fit's prior
  base_delta_source           "prior" when a base delta applies, else None
  sd                          posterior standard deviation per parameter, shaped like
                              shrinkage_weight, plus residual_s: sqrt(diag(s2 (X'X + D)^-1))
                              with D the penalty (0 unpenalised, the evolution ridge on its
                              columns) and s2 = RSS / (n - p) over the fit's rows, p its
                              non-zero columns. A shrunk fit with no residual degrees of
                              freedom takes s2 from the prior. A parameter held at its prior
                              (weight 0) reports the prior tau instead: the shrinkage tau, and
                              for a fixed fuel slope `fuel_prior_sd` (NaN when not given).
                              NaN wherever s2 cannot be estimated.

`fit(..., base_delta=x)` adds x seconds per lap to the base of every compound fitted on
practice rows only (no race rows): the practice-to-race level difference (practice runs
start on less fuel than the race, plus evolution and temperature). It is a config prior
for now, and the first factor SPEC H's calibrator will predict per race. Under
to_snapshot it sits inside the prior's bases, so the race moves it with base_s tau.

Practice rows with session and stint_start_min get one slope per session in the stint's
start time, fitted with base and wear in the ordinary fits with a ridge toward 0
(EVOLUTION_TAU_S_PER_MIN), so track evolution between runs is not read as tyre wear.
The term is constant within a stint, so it is identified across stints only.
"""
import numpy as np

from .data import EVOLUTION_TAU_S_PER_MIN, evolution_columns, ridge_lstsq
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


def _evolution_prec(n_cols, n_evo):
    """The ridge precision of _evolution_fit: on the last n_evo (evolution) columns only."""
    prec = np.zeros(n_cols)
    if n_evo:
        prec[-n_evo:] = EVOLUTION_TAU_S_PER_MIN ** -2
    return prec


def _evolution_fit(X, y, n_evo):
    """Least squares with the ridge on the last n_evo (evolution) columns."""
    return ridge_lstsq(X, y, _evolution_prec(X.shape[1], n_evo))


def _posterior_sd(X, y, beta, prec, fallback_s2=float("nan")):
    """(sd per column, residual sd): sqrt(diag(s2 (X'X + D)^-1)), D = diag(prec).

    s2 = RSS / (n - p) over the rows of X, p the non-zero columns; `fallback_s2` when
    n <= p. NaN where X'X + D is singular.
    """
    n, k = X.shape
    used = int(np.any(X != 0, axis=0).sum()) if X.size else 0
    if n > used:
        r = y - X @ beta
        s2 = float(r @ r) / (n - used)
    else:
        s2 = float(fallback_s2)
    if k == 0:
        return np.empty(0), float(np.sqrt(s2))
    try:
        cov = np.linalg.inv(X.T @ X + np.diag(prec))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), float(np.sqrt(s2))
    return np.sqrt(np.clip(s2 * np.diag(cov), 0.0, None)), float(np.sqrt(s2))


def _unmodelled(data, comps, min_samples, support=None):
    """Compounds in `data` (or unmodelled by the prior) that are not modelled, with why."""
    out = {}
    for c in sorted(set(data["compound"].dropna()) - set(comps)):
        rows = data[data["compound"] == c]
        if "run_class" in rows and not (rows["run_class"] == "long_run").any():
            out[c] = "no long run"
        else:
            out[c] = f"{int(rows['clean'].sum())} clean laps, under min_samples {min_samples}"
    for c, why in ((support or {}).get("unmodelled") or {}).items():
        if c not in comps:
            out.setdefault(c, why)
    return out


def _report(d, comps, X, fuel_slope_source, weight, has_fuel, support=None, evolution=None, unmodelled=None,
            sd=None, residual_s=float("nan")):
    """`weight` and `sd` are per parameter: base and wear per compound, then fuel when has_fuel.
    `support` is the prior's fit_report, whose rows count toward n_rows and max_tyre_age."""
    sd = np.full(len(weight), np.nan) if sd is None else sd
    race = ~np.isnan(_race_lap(d))
    support = support or {"n_rows": {}, "max_tyre_age": {}}
    rep = {"n_rows": {}, "n_race_rows": int(race.sum()), "n_race_rows_per_compound": {},
           "max_tyre_age": {}, "fuel_slope_source": fuel_slope_source, "condition_number": _cond(X),
           "shrinkage_weight": {"base_s": {}, "wear_s_per_lap": {},
                                "fuel_s_per_lap": float(weight[-1]) if has_fuel else None},
           "evolution_s_per_min": dict(evolution or {}), "unmodelled": dict(unmodelled or {}),
           "sd": {"base_s": {}, "wear_s_per_lap": {}, "fuel_s_per_lap": float(sd[-1]) if has_fuel else None,
                  "residual_s": float(residual_s)}}
    for i, c in enumerate(comps):
        m = (d["compound"] == c).to_numpy()
        ages = [float(d.loc[m, "tyre_age"].max())] if m.any() else []
        ages += [support["max_tyre_age"][c]] if c in support["max_tyre_age"] else []
        rep["n_rows"][c] = int(m.sum()) + support["n_rows"].get(c, 0)
        rep["n_race_rows_per_compound"][c] = int((m & race).sum())
        rep["max_tyre_age"][c] = max(ages) if ages else float("nan")
        rep["shrinkage_weight"]["base_s"][c] = float(weight[2 * i])
        rep["shrinkage_weight"]["wear_s_per_lap"][c] = float(weight[2 * i + 1])
        rep["sd"]["base_s"][c] = float(sd[2 * i])
        rep["sd"]["wear_s_per_lap"][c] = float(sd[2 * i + 1])
    # a shrunk fit carries the base delta inside the prior's bases
    rep["base_delta_s"] = {c: v for c, v in (support.get("base_delta_s") or {}).items() if c in comps}
    rep["base_delta_source"] = "prior" if rep["base_delta_s"] else None
    return rep


def _apply_base_delta(coefs, rep, base_delta):
    """Add the practice-to-race base delta to the compounds fitted on practice rows only."""
    practice = [c for c in coefs if rep["n_race_rows_per_compound"].get(c, 0) == 0]
    rep["base_delta_s"] = {c: float(base_delta) for c in practice}
    rep["base_delta_source"] = "prior" if practice else None
    return {c: (b + base_delta, w) if c in practice else (b, w) for c, (b, w) in coefs.items()}


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
    y = d["lap_time_s"].to_numpy(dtype=float)
    if cols:
        rhs = X.T @ y + prec * beta0
        beta = np.linalg.lstsq(xtx + np.diag(prec), rhs, rcond=None)[0]
    else:
        beta = beta0
    dj = np.diag(xtx)
    weight = np.divide(dj, dj + prec, out=np.ones_like(dj), where=prec > 0)
    source = None if fuel_prior is None else ("shrunk" if len(d) else "prior")
    support = prior.fit_report if prior is not None else None
    prior_res = ((support or {}).get("sd") or {}).get("residual_s", float("nan"))
    sd, res = _posterior_sd(X, y, beta, prec, fallback_s2=prior_res ** 2)
    tau = np.divide(1.0, np.sqrt(prec), out=np.full_like(prec, np.nan), where=prec > 0)
    sd = np.where((weight <= 0) & (prec > 0), tau, sd)  # held at the prior: its tau
    return comps, beta, _report(d, comps, X, source, weight, fuel_prior is not None, support,
                                unmodelled=_unmodelled(data, comps, min_samples, support), sd=sd, residual_s=res)


def _coefs(comps, beta):
    return {c: (float(beta[2 * i]), float(beta[2 * i + 1])) for i, c in enumerate(comps)}


@DEGRADATION.register("linear")
class LinearDegradation:
    """Per compound line of lap time against tyre age. Matches the baseline fit."""

    def __init__(self, coefs, fit_report=None):
        self.coefs = coefs  # compound -> (base_s, slope_s_per_lap)
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, prior=None, shrinkage=None, base_delta=0.0, **_):
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
        sessions, evo = evolution_columns(d)
        X = np.column_stack(_compound_columns(d, comps) + evo) if comps else np.empty((0, 0))
        y = d["lap_time_s"].to_numpy(dtype=float)
        evolution = {}
        if comps and evo:  # practice: fit base and wear jointly with the session slopes
            beta = _evolution_fit(X, y, len(evo))
            coefs = _coefs(comps, beta)
            evolution = {s: float(b) for s, b in zip(sessions, beta[2 * len(comps):])}
        else:  # the per compound lines are the joint least squares fit on X
            beta = np.array([v for c in comps for v in coefs[c]], dtype=float)
        sd, res = (_posterior_sd(X, y, beta, _evolution_prec(X.shape[1], len(evo))) if comps
                   else (np.empty(0), float("nan")))
        rep = _report(d, comps, X, None, np.ones(2 * len(comps)), False, evolution=evolution,
                      unmodelled=_unmodelled(data, comps, min_samples), sd=sd[:2 * len(comps)], residual_s=res)
        return cls(_apply_base_delta(coefs, rep, base_delta), rep)

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
    to the lap column; their in-stint fuel, `fuel_prior` x lap_in_stint, is taken off
    their lap times. When no compound has min_samples race rows the lap term cannot be
    identified, so the slope is fixed to `fuel_prior` (SimConfig.pre_race_fuel_slope_s_per_lap)
    and base and wear are fitted on lap times with the prior's contribution taken off.
    With `shrinkage`, the penalised fit toward `prior` described in the module docstring.
    `fuel_prior_sd` (SimConfig.shrinkage fuel tau) is the fit_report sd of a fixed fuel
    slope; NaN when not given.
    """

    def __init__(self, coefs, fuel_slope, fit_report=None):
        self.coefs = coefs
        self.fuel_slope = fuel_slope  # seconds per race lap, expected negative
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, fuel_prior=-0.045, prior=None, shrinkage=None, base_delta=0.0,
            fuel_prior_sd=None, **_):
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage, fuel_prior)
            return cls(_coefs(comps, beta), float(beta[-1]), rep)
        fixed_sd = float("nan") if fuel_prior_sd is None else float(fuel_prior_sd)
        comps = [c for c in sorted(data["compound"].unique()) if _usable(data, c, min_samples) is not None]
        d = data[data["compound"].isin(comps) & data["clean"]]
        unmodelled = _unmodelled(data, comps, min_samples)
        if not comps:
            return cls({}, fuel_prior, _report(d, comps, np.empty((0, 0)), "prior", np.zeros(1), True,
                                               unmodelled=unmodelled, sd=np.array([fixed_sd])))
        lap = _race_lap(d)
        race = ~np.isnan(lap)
        lap = np.where(race, lap, 0.0)
        y = d["lap_time_s"].to_numpy(dtype=float)
        if "lap_in_stint" in d:  # practice: in-stint fuel column, coefficient fixed to the prior
            run = d["lap_in_stint"].to_numpy(dtype=float)
            y = y - np.where(~race & ~np.isnan(run), fuel_prior * np.nan_to_num(run), 0.0)
        n_race = [int(((d["compound"] == c).to_numpy() & race).sum()) for c in comps]
        cols = _compound_columns(d, comps)
        sessions, evo = evolution_columns(d)
        k = 2 * len(comps)
        if any(n >= min_samples for n in n_race):
            X = np.column_stack(cols + [lap] + evo)
            beta = _evolution_fit(X, y, len(evo))
            fuel, source, slopes = float(beta[k]), "fitted", beta[k + 1:]
            sd, res = _posterior_sd(X, y, beta, _evolution_prec(X.shape[1], len(evo)))
            sd = sd[:k + 1]
        else:
            X = np.column_stack(cols + evo)
            beta = _evolution_fit(X, y - fuel_prior * lap, len(evo))
            fuel, source, slopes = float(fuel_prior), "prior", beta[k:]
            sd, res = _posterior_sd(X, y - fuel_prior * lap, beta, _evolution_prec(X.shape[1], len(evo)))
            sd = np.append(sd[:k], fixed_sd)  # the fuel slope is fixed at the prior: its tau
        weight = np.append(np.ones(k), 1.0 if source == "fitted" else 0.0)
        evolution = {s: float(b) for s, b in zip(sessions, slopes)}
        rep = _report(d, comps, X, source, weight, True, evolution=evolution, unmodelled=unmodelled,
                      sd=sd, residual_s=res)
        return cls(_apply_base_delta(_coefs(comps, beta), rep, base_delta), fuel, rep)

    @property
    def compounds(self):
        return set(self.coefs)

    def draw(self, rng):
        return self

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float) + self.fuel_slope * np.asarray(laps, dtype=float)
