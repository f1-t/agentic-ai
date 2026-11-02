"""Lap time models. A model maps (compound, tyre age, lap number) to a green-flag lap time.

`draw(rng)` returns a concrete lap time function for one sample. Deterministic models
(and the linear ones under the default draw "none") return themselves, uncertainty-aware
models return a parameter draw.

Rows with `lap` NaN are practice laps: they count for compound base and tyre wear only,
never for the race lap (fuel) term. Inside a practice run tyre age and fuel burn advance
together, so practice alone cannot tell wear from the fuel gain: linear_fuel gives
practice rows a fuel column, lap_in_stint, with its coefficient fixed to `fuel_prior`
(the prior's part is taken off their lap times before fitting).

linear_fuel_hinge is linear_fuel plus a per compound hinge max(0, age - knot), the
robustness variant to linear_fuel (the default); its hinge rides every fit path below.

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
  hinge                       linear_fuel_hinge only: {compound: {value, sd, source, weight,
                              knot, n_past_knot}}, see LinearFuelHingeDegradation
  cov                         the posterior covariance, {"names": [...], "matrix": [[...]]},
                              s2 (X'X + D)^-1 over the parameters in a fixed order:
                              base_s[c], wear_s_per_lap[c] per compound (sorted), then
                              fuel_s_per_lap, then hinge_s_per_lap[c] per hinged compound; the
                              evolution slopes are left out. A parameter held or fixed gets
                              its prior variance on the diagonal and no covariance: a fixed
                              fuel slope fuel_prior_sd^2, a weight-0 shrunk coefficient tau^2;
                              a dropped hinge is left out (fixed at 0, variance 0). NaN
                              matrix when X'X + D is singular. A
                              held prior (calibrators.held_prior) carries the calibrator's
                              factor sds instead.

`draw(rng)`: SimConfig.degradation.params `draw` "none" (the default: the point estimate,
nothing drawn) or "posterior" (beta_hat + L z from fit_report["cov"], _ParameterDraws);
`clip_wear_at_zero` (default false) floors drawn wear slopes at 0.

`wear_lower_bound` (SimConfig.degradation.params, default None: no bound): a tyre does not
get faster with age. With a value, every fit (full, practice, to_snapshot; linear,
linear_fuel, linear_fuel_hinge) solves its least squares or penalised least squares with
wear_s_per_lap[c] >= the bound for every compound, by an active-set loop over the wear
coordinates (_active_set; hinges and the other parameters are not constrained). A fit
whose unconstrained wear is all at or above the bound is the unbounded fit, bit for bit.
fit_report then gains:
  bounded                     {compound: True when its wear sits on the bound}
  unconstrained               only when a wear sits on the bound: {"names", "mean", "matrix"},
                              the unconstrained fit's mean and cov, in cov order. sd and
                              cov are then the free parameters' conditional posterior given
                              the bounded ones (their rows and columns dropped from X'X + D
                              before inverting, s2 from the constrained residuals); a bounded
                              wear keeps its unconstrained variance, with no covariance.
  draw_stats                  with draw "posterior": {n_samples, n_proposals, n_clipped,
                              acceptance_rate}, counted over this fit's draws
The point estimate is the constrained mode. The posterior draw is the unconstrained
Gaussian (fit_report["unconstrained"], else the cov around the point estimate) truncated
to wear >= the bound: proposals are drawn jointly and a draw with any wear below the
bound is rejected, up to DRAW_TRIES per sample, the last one clipped at the bound (and
counted) when all fail. Rejection keeps the joint correlation: a rejected negative wear
takes its fuel partner with it. The clip fallback biases the bounded wear draws low: where
the unconstrained mean sits far below the bound most samples clip to it (Spain to_snapshot
k 6-11: up to 122 of 200), while the truncated normal's mean is above it (about 0.37 sd
above the bound for a mean 2 sd below), so the drawn wear is low by up to about 0.4 sd.
draw_stats reports how often it happens; an exact one-dimensional truncated-normal draw is
the refinement if it matters.

Two to_snapshot options (SimConfig.degradation.params, both off by default; cross-race arms,
SPEC 0016): `divide_data_by_s2` (the prior-sd fix of SPEC 0017 G) divides the data term by
the fit's residual variance, so each tau is a prior sd in seconds; `collinear_hold` (a
threshold, e.g. 0.5) holds a compound's wear at the prior while |corr(fuel, wear[c])| of
the shrinkage fit is above it. See _shrunk.

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
import copy
import warnings

import numpy as np

from .config import HINGE_KNOT, HINGE_TAU_S_PER_LAP
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
    """(sd per column, residual sd, cov): cov = s2 (X'X + D)^-1, D = diag(prec), and
    sd = sqrt(diag(cov)).

    s2 = RSS / (n - p) over the rows of X, p the non-zero columns; `fallback_s2` when
    n <= p. NaN (sd and the whole matrix) where X'X + D is singular.
    """
    n, k = X.shape
    used = int(np.any(X != 0, axis=0).sum()) if X.size else 0
    if n > used:
        r = y - X @ beta
        s2 = float(r @ r) / (n - used)
    else:
        s2 = float(fallback_s2)
    if k == 0:
        return np.empty(0), float(np.sqrt(s2)), np.empty((0, 0))
    try:
        inv = np.linalg.inv(X.T @ X + np.diag(prec))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), float(np.sqrt(s2)), np.full((k, k), np.nan)
    return np.sqrt(np.clip(s2 * np.diag(inv), 0.0, None)), float(np.sqrt(s2)), s2 * inv


def _solve_fixed(X, y, prec, b0, fixed, bound, ridge):
    """The minimiser of ||y - X b||^2 + sum_j prec_j (b_j - b0_j)^2 with b_j = bound on the
    `fixed` columns: their columns times the bound move to the right-hand side and the rest
    are solved as the unconstrained fit solves them (`ridge`: ridge_lstsq, b0 0; else the
    normal equations). Returns (b, the reduced X, y, prec)."""
    free = ~fixed
    Xu, yu, pu = X[:, free], y - X[:, fixed] @ np.full(int(fixed.sum()), float(bound)), prec[free]
    if ridge:
        bu = ridge_lstsq(Xu, yu, pu)
    else:
        bu = np.linalg.lstsq(Xu.T @ Xu + np.diag(pu), Xu.T @ yu + pu * b0[free], rcond=None)[0]
    b = np.full(X.shape[1], float(bound))
    b[free] = bu
    return b, Xu, yu, pu


def _objective(X, y, prec, b0, b):
    r = y - X @ b
    return float(r @ r + np.sum(prec * (b - b0) ** 2))


def _active_set(X, y, prec, b0, beta, wear, bound, ridge):
    """(b, fixed): min ||y - X b||^2 + sum_j prec_j (b_j - b0_j)^2 subject to b_j >= bound on
    the `wear` columns, from the unconstrained solution `beta`. Fix every wear coordinate
    below the bound at it and re-solve; release a fixed one whose gradient says the objective
    falls inside the bound, re-solve; at most 2^n_wear passes. A loop that has not settled
    by then (it does not happen on these problems) takes the best feasible fixed set by
    brute force. `beta` is returned as it is when it is feasible."""
    fixed = np.zeros(len(beta), dtype=bool)
    if not np.any(beta[wear] < bound):
        return beta, fixed
    H = X.T @ X + np.diag(prec)
    rhs = X.T @ y + prec * b0
    b = beta
    for _ in range(2 ** int(wear.sum())):
        low = wear & ~fixed & (b < bound)
        if low.any():
            fixed |= low
        else:
            grad = H @ b - rhs   # half the gradient; < 0 means the objective falls as b_j rises
            release = fixed & (grad < -1e-9 * (1.0 + np.abs(rhs)))
            if not release.any():
                return b, fixed
            fixed &= ~release
        b = _solve_fixed(X, y, prec, b0, fixed, bound, ridge)[0]
    best = None
    idx = np.flatnonzero(wear)
    for mask in range(2 ** len(idx)):
        f = np.zeros(len(beta), dtype=bool)
        f[idx[[bool(mask >> i & 1) for i in range(len(idx))]]] = True
        bb = _solve_fixed(X, y, prec, b0, f, bound, ridge)[0]
        if np.all(bb[wear] >= bound):
            obj = _objective(X, y, prec, b0, bb)
            if best is None or obj < best[0]:
                best = (obj, bb, f)
    return best[1], best[2]


def _bound_fit(X, y, prec, b0, beta, posterior, n_comps, bound, ridge, fallback_s2=float("nan"), wear=None):
    """wear_lower_bound on one fit. `posterior` is _posterior_sd's (sd, residual sd, cov) at
    the unconstrained `beta`; the wear columns are 1, 3, ... 2 n_comps - 1, or the `wear`
    mask when given (a fit with some columns held, collinear_hold).

    Returns (beta, fixed, (sd, residual sd, cov), unconstrained): with nothing at the bound,
    `beta` and `posterior` as given and unconstrained None. Otherwise the constrained mode;
    the free parameters' conditional posterior given the bounded ones (the bounded columns
    dropped before inverting, s2 from the constrained residuals), the bounded wear keeping
    its unconstrained variance with no covariance; and unconstrained = (unconstrained beta
    less the constrained one, unconstrained cov), the Gaussian the posterior draws truncate."""
    if wear is None:
        wear = np.zeros(X.shape[1], dtype=bool)
        wear[1:2 * n_comps:2] = True
    b, fixed = _active_set(X, y, prec, b0, beta, wear, bound, ridge)
    if not fixed.any():
        return beta, fixed, posterior, None
    _, Xu, yu, pu = _solve_fixed(X, y, prec, b0, fixed, bound, ridge)
    sd_u, res, cov_u = _posterior_sd(Xu, yu, b[~fixed], pu, fallback_s2)
    sd0, _, cov0 = posterior
    sd, cov = np.array(sd0, dtype=float), np.zeros_like(cov0)
    free = np.flatnonzero(~fixed)
    sd[free] = sd_u
    cov[np.ix_(free, free)] = cov_u
    for j in np.flatnonzero(fixed):
        cov[j, j] = cov0[j, j]
    return b, fixed, (sd, res, cov), (beta - b, cov0)


def _bound_report(rep, comps, fixed, names, cols, unconstrained, fixed_var=None):
    """fit_report["bounded"] = {compound: wear on the bound}; with a wear on it, also
    fit_report["unconstrained"] = {"names", "shift", "matrix"}: the unconstrained posterior
    over `names` (cov order), its mean the point estimate plus `shift` (with_draw turns it
    into "mean"). `cols` names the fit's columns, `fixed_var` as _place_cov's `fixed`."""
    rep["bounded"] = {c: bool(fixed[2 * i + 1]) for i, c in enumerate(comps)}
    if unconstrained is not None:
        shift, cov = unconstrained
        at = {n: j for j, n in enumerate(cols)}
        rep["unconstrained"] = {"names": list(names),
                                "shift": [float(shift[at[n]]) if n in at else 0.0 for n in names],
                                "matrix": _place_cov(names, cols, cov, fixed_var)["matrix"]}


def _param_names(comps, has_fuel, hinged=None):
    """fit_report["cov"] names in their fixed order: base_s[c], wear_s_per_lap[c] per compound
    (sorted), fuel_s_per_lap, hinge_s_per_lap[c] per hinged compound (a dropped hinge is not
    a parameter: it stays at 0, variance 0)."""
    names = [f"{p}[{c}]" for c in sorted(comps) for p in ("base_s", "wear_s_per_lap")]
    names += ["fuel_s_per_lap"] if has_fuel else []
    return names + [f"hinge_s_per_lap[{c}]" for c in sorted(hinged or [])]


def _cov_entry(names, matrix):
    return {"names": list(names), "matrix": [[float(v) for v in row] for row in np.asarray(matrix, dtype=float)]}


def _place_cov(names, cols, cov, fixed=None):
    """The fit_report["cov"] entry over `names`: `cols` names the rows of `cov` (columns of
    the fit; names not in `names`, the evolution slopes, are left out); every name not in
    `cols` gets the variance in `fixed` (0 when absent) and zero covariance."""
    m = np.zeros((len(names), len(names)))
    at = {n: i for i, n in enumerate(names)}
    keep = [(j, at[c]) for j, c in enumerate(cols) if c in at]
    if keep:
        src, dst = map(list, zip(*keep))
        m[np.ix_(dst, dst)] = np.asarray(cov)[np.ix_(src, src)]
    for n, var in (fixed or {}).items():
        i = at[n]
        m[i, :] = m[:, i] = 0.0
        m[i, i] = var
    return _cov_entry(names, m)


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


def _knots(comps, hinge_knots, hinge_knot_default):
    """compound -> hinge knot: SimConfig.hinge_knots, else hinge_knot_default."""
    return {c: int((hinge_knots or {}).get(c, hinge_knot_default)) for c in comps}


def _hinge_column(d, compound, knot):
    m = (d["compound"] == compound).to_numpy(dtype=float)
    return m * np.maximum(0.0, d["tyre_age"].to_numpy(dtype=float) - knot)


def _past_knot(d, compound, knot):
    """Rows of `compound` in `d` with tyre age past `knot`."""
    return int(((d["compound"] == compound).to_numpy() & (d["tyre_age"].to_numpy(dtype=float) > knot)).sum())


def _hinge_entry(value, sd, source, weight, knot, n_past):
    return {"value": float(value), "sd": float(sd), "source": source, "weight": float(weight),
            "knot": int(knot), "n_past_knot": int(n_past)}


def _dropped(knot, n_past):
    """A hinge with too few rows past its knot: fixed at 0, no sd."""
    return _hinge_entry(0.0, float("nan"), "dropped", 0.0, knot, n_past)


def _fuel_wear_corr(cov, n_comps, fuel_col):
    """corr(fuel, wear[c]) per compound index from a fit's cov, NaN where a variance is not
    above 0 or a value is NaN (model_report.fuel_wear_corr's rule, on the fit's columns)."""
    out = np.full(n_comps, np.nan)
    vf = cov[fuel_col, fuel_col]
    for i in range(n_comps):
        j = 2 * i + 1
        vw, c = cov[j, j], cov[fuel_col, j]
        if not np.isnan([vf, vw, c]).any() and vf > 0 and vw > 0:
            out[i] = c / np.sqrt(vf * vw)
    return out


def _hold_columns(X, y, prec, beta0, hold, fallback_s2, n_comps, bound=None):
    """The shrinkage fit with the `hold` columns fixed at their prior means beta0: those columns
    times beta0 move to the right-hand side and the rest are solved as _shrunk solves them,
    under wear >= `bound` on the wear columns left free when a bound is given (_bound_fit; a
    held wear is the prior's, the bound only applies once a compound is released).

    Returns (beta, sd, residual sd, cov, fixed, unconstrained) over every column: the held
    ones at beta0 with sd 0 and no covariance (the caller reports their prior tau); fixed,
    the wear columns on the bound (all False without one); unconstrained, None or
    (shift, cov) in full columns as _bound_fit gives it, 0 on the held columns."""
    free = ~hold
    Xu, pu = X[:, free], prec[free]
    yu = y - X[:, hold] @ beta0[hold]
    bu = np.linalg.lstsq(Xu.T @ Xu + np.diag(pu), Xu.T @ yu + pu * beta0[free], rcond=None)[0]
    sd_u, res, cov_u = _posterior_sd(Xu, yu, bu, pu, fallback_s2)
    k = len(beta0)
    fixed, unconstrained = np.zeros(k, dtype=bool), None
    idx = np.flatnonzero(free)
    if bound is not None:
        wear = np.zeros(k, dtype=bool)
        wear[1:2 * n_comps:2] = True
        bu, fixed_u, (sd_u, res, cov_u), unc = _bound_fit(Xu, yu, pu, beta0[free], bu, (sd_u, res, cov_u), n_comps,
                                                          bound, False, fallback_s2, wear=wear[free])
        fixed[idx] = fixed_u
        if unc is not None:
            shift, cov0 = np.zeros(k), np.zeros((k, k))
            shift[idx] = unc[0]
            cov0[np.ix_(idx, idx)] = unc[1]
            unconstrained = (shift, cov0)
    beta, sd, cov = beta0.copy(), np.zeros(k), np.zeros((k, k))
    beta[idx], sd[idx] = bu, sd_u
    cov[np.ix_(idx, idx)] = cov_u
    return beta, sd, res, cov, fixed, unconstrained


def _shrunk(data, min_samples, prior, shrinkage, fuel_prior=None, knots=None, wear_lower_bound=None,
            collinear_hold=None, divide_data_by_s2=False):
    """Penalised fit of the race rows toward `prior`. Returns (comps, beta, report).
    `wear_lower_bound`: see the module docstring.

    `divide_data_by_s2` (default False; SPEC 0017 G, the prior-sd fix): minimise
    ||y - X b||^2 / s2 + sum_j ((b_j - b0_j) / tau_j)^2 instead, so each tau is a prior sd in
    seconds whatever the residual variance. s2 is the residual variance of the ordinary fit
    (its fallback, the prior's, when it has no residual degrees of freedom); the penalty is
    multiplied by it and the fit solved once more. fit_report gains "data_term_s2".

    `collinear_hold` (default None: off; a threshold such as 0.5, SPEC 0016 amendment): a
    compound's wear is held at the prior mean, shrinkage weight 0 on wear only, while
    |corr(fuel, wear[c])| of the ordinary fit's cov is above the threshold; the fit is then
    solved again with those wear coefficients fixed, base and fuel shrunk as usual. A held
    wear reports its prior tau as sd and variance, no covariance. Needs a fuel term and a
    prior. With wear_lower_bound the bound applies to the wear left free, so a compound is
    held while collinear and bounded once released. fit_report gains "collinear_hold":
    {threshold, corr: {compound: corr of the ordinary fit}, held: {compound: bool}}.

    Modelled: every compound of the prior, plus race compounds with min_samples clean
    race rows. The fuel column is added when fuel_prior is not None. With `knots`
    ({compound: knot}, linear_fuel_hinge) a hinge column per compound with min_samples
    rows past its knot, the prior's support included, follows; its prior mean is the
    prior's hinge (0 when the prior has none) with tau shrinkage["hinge_s_per_lap"].
    beta holds the hinge coefficients last; report["hinge"] has every compound.
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
    n_main = len(cols)
    support = prior.fit_report if prior is not None else None
    if knots is not None:
        prior_hinge = (support or {}).get("hinge") or {}
        n_past = {c: _past_knot(d, c, knots[c]) + prior_hinge.get(c, {}).get("n_past_knot", 0) for c in comps}
        hinged = [c for c in comps if n_past[c] >= min_samples]
        tau_hinge = shrinkage.get("hinge_s_per_lap", HINGE_TAU_S_PER_LAP)
        for c in hinged:
            cols.append(_hinge_column(d, c, knots[c]))
            beta0.append(float((getattr(prior, "hinges", None) or {}).get(c, 0.0)))
            prec.append(tau_hinge ** -2)
    X = np.column_stack(cols) if cols else np.empty((len(d), 0))
    beta0, prec = np.asarray(beta0, dtype=float), np.asarray(prec, dtype=float)
    tau = np.divide(1.0, np.sqrt(prec), out=np.full_like(prec, np.nan), where=prec > 0)
    xtx = X.T @ X
    y = d["lap_time_s"].to_numpy(dtype=float)

    def solve(p):
        if not cols:
            return beta0
        return np.linalg.lstsq(xtx + np.diag(p), X.T @ y + p * beta0, rcond=None)[0]

    beta = solve(prec)
    source = None if fuel_prior is None else ("shrunk" if len(d) else "prior")
    prior_res = ((support or {}).get("sd") or {}).get("residual_s", float("nan"))
    sd, res, cov = _posterior_sd(X, y, beta, prec, fallback_s2=prior_res ** 2)
    data_s2 = None
    if divide_data_by_s2:
        # the data term over s2 is the penalty times s2: the same minimiser, solved again
        data_s2 = float(res ** 2)
        if np.isfinite(data_s2) and data_s2 > 0:
            prec = prec * data_s2
            beta = solve(prec)
            sd, res, cov = _posterior_sd(X, y, beta, prec, fallback_s2=prior_res ** 2)
    dj = np.diag(xtx)
    weight = np.divide(dj, dj + prec, out=np.ones_like(dj), where=prec > 0)
    hold_rep, bounded = None, False
    if collinear_hold is not None:
        if fuel_prior is None or not pc:
            raise ValueError("collinear_hold needs a fuel term and a practice prior")
        corr = _fuel_wear_corr(cov, len(comps), 2 * len(comps))
        hold_c = np.abs(np.nan_to_num(corr)) > float(collinear_hold)
        hold_rep = {"threshold": float(collinear_hold), "corr": {c: float(r) for c, r in zip(comps, corr)},
                    "held": {c: bool(h) for c, h in zip(comps, hold_c)}}
        if hold_c.any():
            hold = np.zeros(len(beta0), dtype=bool)
            hold[2 * np.flatnonzero(hold_c) + 1] = True
            beta, sd, res, cov, fixed, unconstrained = _hold_columns(X, y, prec, beta0, hold, prior_res ** 2,
                                                                     len(comps), wear_lower_bound)
            weight = np.where(hold, 0.0, weight)
            bounded = wear_lower_bound is not None
    if wear_lower_bound is not None and not bounded:
        beta, fixed, (sd, res, cov), unconstrained = _bound_fit(
            X, y, prec, beta0, beta, (sd, res, cov), len(comps), wear_lower_bound, False, fallback_s2=prior_res ** 2)
    held = (weight <= 0) & (prec > 0)
    sd = np.where(held, tau, sd)  # held at the prior: its tau
    rep = _report(d, comps, X, source, weight[:n_main], fuel_prior is not None, support,
                  unmodelled=_unmodelled(data, comps, min_samples, support), sd=sd[:n_main], residual_s=res)
    names = _param_names(comps, fuel_prior is not None, hinged if knots is not None else None)
    rep["cov"] = _place_cov(names, names, cov, {names[j]: tau[j] ** 2 for j in np.flatnonzero(held)})
    if wear_lower_bound is not None:
        _bound_report(rep, comps, fixed, names, names, unconstrained,
                      {names[j]: tau[j] ** 2 for j in np.flatnonzero(held)})
    if knots is not None:
        rep["hinge"] = {c: _dropped(knots[c], n_past[c]) for c in comps}
        for j, c in enumerate(hinged, start=n_main):
            rep["hinge"][c] = _hinge_entry(beta[j], sd[j], "shrunk" if dj[j] > 0 else "prior", weight[j],
                                           knots[c], n_past[c])
    if divide_data_by_s2:
        rep["data_term_s2"] = data_s2
    if hold_rep is not None:
        rep["collinear_hold"] = hold_rep
    return comps, beta, rep


def _coefs(comps, beta):
    return {c: (float(beta[2 * i]), float(beta[2 * i + 1])) for i, c in enumerate(comps)}


PARAM_DRAWS = ("none", "posterior")
DRAW_TRIES = 50   # proposals per sample for a posterior draw under a wear bound


def new_draw_stats():
    """fit_report["draw_stats"] at the start: samples drawn, proposals made, samples that
    fell back to the clipped last proposal, and the acceptance rate, accepted samples over
    proposals (NaN before the first draw)."""
    return {"n_samples": 0, "n_proposals": 0, "n_clipped": 0, "acceptance_rate": float("nan")}


def _split_name(name):
    """'wear_s_per_lap[SOFT]' -> ('wear_s_per_lap', 'SOFT'); 'fuel_s_per_lap' -> ('fuel_s_per_lap', None)."""
    if name.endswith("]") and "[" in name:
        p, _, c = name[:-1].partition("[")
        return p, c
    return name, None


class NaNCovarianceWarning(UserWarning):
    """A fit_report["cov"] row is NaN; posterior draws hold that parameter at its estimate."""


class _ParameterDraws:
    """draw(rng) for the linear models (SimConfig.degradation.params `draw`, `clip_wear_at_zero`).

    "none" (the default) returns the model itself and consumes nothing. "posterior" returns
    a shallow copy whose coefs, fuel_slope and hinges are beta_hat + L z, z standard normal
    from `rng` (draw) or given (draw_z), L the Cholesky factor of fit_report["cov"] (eigh
    with eigenvalues clipped at 0 when it is not positive definite; a parameter whose row
    has a NaN gets 0 variance, with one NaNCovarianceWarning per fitted model).
    clip_wear_at_zero floors drawn wear slopes at 0. The engine passes row k of its
    per-batch DEGRADATION_STREAM normals to draw_z.
    """

    param_draw = "none"
    clip_wear_at_zero = False
    wear_lower_bound = None

    def with_draw(self, draw="none", clip_wear_at_zero=False, wear_lower_bound=None):
        """Set the draw; with wear_lower_bound, finish fit_report["unconstrained"] (its mean,
        the point estimate plus the fit's shift) and, for the posterior draw, start
        fit_report["draw_stats"]."""
        if draw not in PARAM_DRAWS:
            raise ValueError(f"degradation draw must be one of {PARAM_DRAWS}, got {draw!r}")
        self.param_draw, self.clip_wear_at_zero = draw, bool(clip_wear_at_zero)
        if wear_lower_bound is not None:
            self.wear_lower_bound = float(wear_lower_bound)
            unc = self.fit_report.get("unconstrained")
            if unc is not None and "shift" in unc:
                mean = self.param_vector()[1] + np.asarray(unc["shift"], dtype=float)
                self.fit_report["unconstrained"] = {"names": unc["names"], "mean": mean.tolist(),
                                                    "matrix": unc["matrix"]}
            if draw == "posterior":
                self.fit_report["draw_stats"] = new_draw_stats()
        return self

    @property
    def draw_tries(self):
        """Proposals per sample a posterior draw may take: DRAW_TRIES under a wear bound, else 1."""
        return DRAW_TRIES if self.wear_lower_bound is not None and self.param_draw == "posterior" else 1

    def _value(self, name):
        p, c = _split_name(name)
        if p == "fuel_s_per_lap":
            return float(self.fuel_slope)
        if p == "hinge_s_per_lap":
            return float(self.hinges[c])
        return float(self.coefs[c][0 if p == "base_s" else 1])

    def param_vector(self):
        """(names, values) in fit_report["cov"] order."""
        names = self.fit_report["cov"]["names"]
        return names, np.array([self._value(n) for n in names], dtype=float)

    def _plan(self):
        """(L, mean, wear mask, layout), cached on this instance against its cov entry: L with
        L L' = fit_report["cov"], the point estimate in cov order, and where each name goes.
        With fit_report["unconstrained"] (a wear on its bound) its matrix and mean instead."""
        cov = self.fit_report.get("unconstrained") or self.fit_report["cov"]
        cached = getattr(self, "_draw_plan", None)
        if cached is not None and cached[0] is cov and cached[1] == id(self):
            return cached[2]
        names = cov["names"]
        m = np.array(cov["matrix"], dtype=float).reshape(len(names), len(names))
        bad = np.isnan(m).any(axis=1) | np.isnan(m).any(axis=0)
        if bad.any():
            warnings.warn(f"NaN posterior covariance for {[n for n, b in zip(names, bad) if b]}: "
                          "drawn at their estimate", NaNCovarianceWarning, stacklevel=4)
            m[bad, :] = 0.0
            m[:, bad] = 0.0
        try:
            L = np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh((m + m.T) / 2)
            L = v * np.sqrt(np.clip(w, 0.0, None))
        at = {n: i for i, n in enumerate(names)}
        coefs = [(c, at.get(f"base_s[{c}]"), at.get(f"wear_s_per_lap[{c}]"), b, w) for c, (b, w) in self.coefs.items()]
        hinges = getattr(self, "hinges", None)
        hinges = None if hinges is None else [(c, at.get(f"hinge_s_per_lap[{c}]"), h) for c, h in hinges.items()]
        mean = np.array(cov["mean"], dtype=float) if "mean" in cov else self.param_vector()[1]
        plan = (L, mean, np.array([n.startswith("wear_s_per_lap[") for n in names], dtype=bool),
                coefs, at.get("fuel_s_per_lap"), hinges)
        self._draw_plan = (cov, id(self), plan)
        return plan

    @property
    def n_params(self):
        """Length of the parameter vector a posterior draw takes (fit_report["cov"] names)."""
        return len(self.fit_report["cov"]["names"])

    def draw(self, rng):
        if self.param_draw == "none":
            return self
        return self.draw_z(rng.standard_normal(self.n_params * self.draw_tries))

    def draw_z(self, z):
        """The posterior draw for standard normals `z` (n_params of them): beta_hat + L z.
        The engine passes row k of engine.batch_normals for sample k.

        Under a wear bound `z` holds up to draw_tries proposals of n_params each (the engine
        appends DEGRADATION_RETRY_STREAM's row): the first whose wear slopes are all at or
        above the bound is taken, else the last one with its wear clipped at the bound,
        counted in fit_report["draw_stats"]."""
        if self.param_draw == "none":
            return self
        L, mean, wear, coefs, fuel, hinges = self._plan()
        if self.wear_lower_bound is None:
            x = (mean + L @ np.asarray(z, dtype=float)).tolist()
        else:
            x = self._truncated(L, mean, wear, np.asarray(z, dtype=float))
        if self.clip_wear_at_zero:
            x = [max(v, 0.0) if w else v for v, w in zip(x, wear)]
        out = copy.copy(self)
        out.coefs = {c: (b if i is None else x[i], w if j is None else x[j]) for c, i, j, b, w in coefs}
        if fuel is not None:
            out.fuel_slope = x[fuel]
        if hinges is not None:
            out.hinges = {c: h if i is None else x[i] for c, i, h in hinges}
        return out

    def _truncated(self, L, mean, wear, z):
        """One draw from N(mean, L L') truncated to wear >= the bound, by rejection over the
        proposals in `z`; the clipped last proposal when every one is rejected (biased low by
        up to about 0.4 sd where that is common, see the module docstring)."""
        bound = self.wear_lower_bound
        rows = z.reshape(-1, len(mean))
        clipped = True
        for tries, zi in enumerate(rows, start=1):
            x = mean + L @ zi   # as the unbounded draw computes it, so an accepted first one matches it
            if np.all(x[wear] >= bound):
                clipped = False
                break
        if clipped:
            x = np.where(wear, np.maximum(x, bound), x)
        stats = self.fit_report.get("draw_stats")
        if stats is not None:
            stats["n_samples"] += 1
            stats["n_proposals"] += tries
            stats["n_clipped"] += int(clipped)
            stats["acceptance_rate"] = (stats["n_samples"] - stats["n_clipped"]) / stats["n_proposals"]
        return x.tolist()


@DEGRADATION.register("linear")
class LinearDegradation(_ParameterDraws):
    """Per compound line of lap time against tyre age. Matches the baseline fit."""

    def __init__(self, coefs, fit_report=None):
        self.coefs = coefs  # compound -> (base_s, slope_s_per_lap)
        self.fit_report = fit_report

    @classmethod
    def fit(cls, data, min_samples=5, prior=None, shrinkage=None, base_delta=0.0, draw="none",
            clip_wear_at_zero=False, wear_lower_bound=None, divide_data_by_s2=False, **_):
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage, wear_lower_bound=wear_lower_bound,
                                       divide_data_by_s2=divide_data_by_s2)
            return cls(_coefs(comps, beta), rep).with_draw(draw, clip_wear_at_zero, wear_lower_bound)
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
        sd, res, cov = (_posterior_sd(X, y, beta, _evolution_prec(X.shape[1], len(evo))) if comps
                        else (np.empty(0), float("nan"), np.empty((0, 0))))
        fixed, unconstrained = np.zeros(len(beta), dtype=bool), None
        if wear_lower_bound is not None and comps:
            prec = _evolution_prec(X.shape[1], len(evo))
            beta, fixed, (sd, res, cov), unconstrained = _bound_fit(
                X, y, prec, np.zeros(len(prec)), beta, (sd, res, cov), len(comps), wear_lower_bound, True)
            if fixed.any():
                coefs = _coefs(comps, beta)
                evolution = {s: float(b) for s, b in zip(sessions, beta[2 * len(comps):])} if evo else {}
        rep = _report(d, comps, X, None, np.ones(2 * len(comps)), False, evolution=evolution,
                      unmodelled=_unmodelled(data, comps, min_samples), sd=sd[:2 * len(comps)], residual_s=res)
        names = _param_names(comps, False)
        rep["cov"] = _place_cov(names, names, cov)
        if wear_lower_bound is not None:
            _bound_report(rep, comps, fixed, names, names, unconstrained)
        return cls(_apply_base_delta(coefs, rep, base_delta), rep).with_draw(draw, clip_wear_at_zero, wear_lower_bound)

    @property
    def compounds(self):
        return set(self.coefs)

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float)


@DEGRADATION.register("linear_fuel")
class LinearFuelDegradation(_ParameterDraws):
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
            fuel_prior_sd=None, draw="none", clip_wear_at_zero=False, wear_lower_bound=None,
            collinear_hold=None, divide_data_by_s2=False, **_):
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage, fuel_prior,
                                       wear_lower_bound=wear_lower_bound, collinear_hold=collinear_hold,
                                       divide_data_by_s2=divide_data_by_s2)
            return cls(_coefs(comps, beta), float(beta[-1]), rep).with_draw(draw, clip_wear_at_zero, wear_lower_bound)
        coefs, fuel, _, rep = _fuel_fit(data, min_samples, fuel_prior, base_delta, fuel_prior_sd,
                                        wear_lower_bound=wear_lower_bound)
        return cls(coefs, fuel, rep).with_draw(draw, clip_wear_at_zero, wear_lower_bound)

    @property
    def compounds(self):
        return set(self.coefs)

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        return base + slope * np.asarray(ages, dtype=float) + self.fuel_slope * np.asarray(laps, dtype=float)


def _fuel_fit(data, min_samples, fuel_prior, base_delta, fuel_prior_sd, hinge_knots=None, wear_lower_bound=None):
    """The unpenalised linear_fuel fit. Returns (coefs, fuel slope, hinges, report).

    With `hinge_knots` ({compound: knot}, linear_fuel_hinge) a hinge column follows the
    fuel column for each compound with min_samples fit rows past its knot; the others are
    dropped (hinge 0). hinges is {compound: s/lap} and the report gains "hinge"; without
    knots hinges is None and the fit is exactly linear_fuel's. `wear_lower_bound`: see the
    module docstring.
    """
    fixed_sd = float("nan") if fuel_prior_sd is None else float(fuel_prior_sd)
    comps = [c for c in sorted(data["compound"].unique()) if _usable(data, c, min_samples) is not None]
    d = data[data["compound"].isin(comps) & data["clean"]]
    unmodelled = _unmodelled(data, comps, min_samples)
    n_past = {c: _past_knot(d, c, hinge_knots[c]) for c in comps} if hinge_knots is not None else {}
    hinged = [c for c in comps if n_past.get(c, 0) >= min_samples]
    if not comps:
        rep = _report(d, comps, np.empty((0, 0)), "prior", np.zeros(1), True,
                      unmodelled=unmodelled, sd=np.array([fixed_sd]))
        rep["cov"] = _cov_entry(["fuel_s_per_lap"], [[fixed_sd ** 2]])
        if wear_lower_bound is not None:
            rep["bounded"] = {}
        if hinge_knots is not None:
            rep["hinge"] = {}
        return {}, fuel_prior, None if hinge_knots is None else {}, rep
    lap = _race_lap(d)
    race = ~np.isnan(lap)
    lap = np.where(race, lap, 0.0)
    y = d["lap_time_s"].to_numpy(dtype=float)
    if "lap_in_stint" in d:  # practice: in-stint fuel column, coefficient fixed to the prior
        run = d["lap_in_stint"].to_numpy(dtype=float)
        y = y - np.where(~race & ~np.isnan(run), fuel_prior * np.nan_to_num(run), 0.0)
    n_race = [int(((d["compound"] == c).to_numpy() & race).sum()) for c in comps]
    cols = _compound_columns(d, comps)
    hcols = [_hinge_column(d, c, hinge_knots[c]) for c in hinged]
    sessions, evo = evolution_columns(d)
    k, h = 2 * len(comps), len(hinged)
    fitted = any(n >= min_samples for n in n_race)
    X = np.column_stack(cols + [lap] + hcols + evo) if fitted else np.column_stack(cols + hcols + evo)
    target = y if fitted else y - fuel_prior * lap
    beta = _evolution_fit(X, target, len(evo))
    sd, res, cov = _posterior_sd(X, target, beta, _evolution_prec(X.shape[1], len(evo)))
    bound_fixed, unconstrained = np.zeros(len(beta), dtype=bool), None
    if wear_lower_bound is not None:
        prec = _evolution_prec(X.shape[1], len(evo))
        beta, bound_fixed, (sd, res, cov), unconstrained = _bound_fit(
            X, target, prec, np.zeros(len(prec)), beta, (sd, res, cov), len(comps), wear_lower_bound, True)
    if fitted:
        fuel, source, slopes = float(beta[k]), "fitted", beta[k + 1 + h:]
        hinge, hinge_sd = beta[k + 1:k + 1 + h], sd[k + 1:k + 1 + h]
        sd = sd[:k + 1]
        colnames, fixed = _param_names(comps, True, hinged), {}
    else:
        fuel, source, slopes = float(fuel_prior), "prior", beta[k + h:]
        hinge, hinge_sd = beta[k:k + h], sd[k:k + h]
        sd = np.append(sd[:k], fixed_sd)  # the fuel slope is fixed at the prior: its tau
        colnames, fixed = _param_names(comps, False, hinged), {"fuel_s_per_lap": fixed_sd ** 2}
    weight = np.append(np.ones(k), 1.0 if source == "fitted" else 0.0)
    evolution = {s: float(b) for s, b in zip(sessions, slopes)}
    rep = _report(d, comps, X, source, weight, True, evolution=evolution, unmodelled=unmodelled,
                  sd=sd, residual_s=res)
    rep["cov"] = _place_cov(_param_names(comps, True, hinged), colnames, cov, fixed)
    if wear_lower_bound is not None:
        _bound_report(rep, comps, bound_fixed, _param_names(comps, True, hinged), colnames, unconstrained, fixed)
    hinges = None
    if hinge_knots is not None:
        rep["hinge"] = {c: _dropped(hinge_knots[c], n_past[c]) for c in comps}
        for c, v, s in zip(hinged, hinge, hinge_sd):
            rep["hinge"][c] = _hinge_entry(v, s, "fitted", 1.0, hinge_knots[c], n_past[c])
        hinges = {c: rep["hinge"][c]["value"] for c in comps}
    return _apply_base_delta(_coefs(comps, beta), rep, base_delta), fuel, hinges, rep


@DEGRADATION.register("linear_fuel_hinge")
class LinearFuelHingeDegradation(LinearFuelDegradation):
    """linear_fuel plus a per compound hinge: lap time = base_c + wear_c * age
    + hinge_c * max(0, age - knot_c) + fuel * lap, so wear may steepen on old tyres.

    Knots come from `hinge_knots` (SimConfig.hinge_knots), else `hinge_knot_default`
    (SimConfig.hinge_knot_default, the filter's HINGE_KNOT). A compound with fewer than
    min_samples fit rows past its knot (practice support included under to_snapshot) has
    its hinge dropped: fixed at 0, source "dropped" in fit_report["hinge"]. Every fit path
    is linear_fuel's with the hinge columns added: unpenalised under full and on practice
    (so a pre_race prior carries the practice hinge), and under to_snapshot shrunk toward
    the prior's hinge (0 when the prior has none) with tau shrinkage["hinge_s_per_lap"].

    fit_report["hinge"]: {compound: {value, sd, source, weight, knot, n_past_knot}}, source
    "fitted", "shrunk", "prior" (no race rows past the knot) or "dropped".
    """

    def __init__(self, coefs, fuel_slope, hinges, knots, fit_report=None):
        super().__init__(coefs, fuel_slope, fit_report)
        self.hinges = hinges  # compound -> s/lap per lap of tyre age past the knot
        self.knots = knots    # compound -> tyre age

    @classmethod
    def fit(cls, data, min_samples=5, fuel_prior=-0.045, prior=None, shrinkage=None, base_delta=0.0,
            fuel_prior_sd=None, hinge_knots=None, hinge_knot_default=HINGE_KNOT, draw="none",
            clip_wear_at_zero=False, wear_lower_bound=None, collinear_hold=None, divide_data_by_s2=False, **_):
        candidates = set(data["compound"].dropna()) | (set(prior.coefs) if prior is not None else set())
        knots = _knots(sorted(candidates), hinge_knots, hinge_knot_default)
        if shrinkage is not None:
            comps, beta, rep = _shrunk(data, min_samples, prior, shrinkage, fuel_prior, knots, wear_lower_bound,
                                       collinear_hold, divide_data_by_s2)
            hinges = {c: rep["hinge"][c]["value"] for c in comps}
            return cls(_coefs(comps, beta), float(beta[2 * len(comps)]), hinges, {c: knots[c] for c in comps},
                       rep).with_draw(draw, clip_wear_at_zero, wear_lower_bound)
        coefs, fuel, hinges, rep = _fuel_fit(data, min_samples, fuel_prior, base_delta, fuel_prior_sd, knots,
                                             wear_lower_bound)
        return cls(coefs, fuel, hinges, {c: knots[c] for c in coefs}, rep).with_draw(draw, clip_wear_at_zero,
                                                                                     wear_lower_bound)

    def lap_times(self, compound, ages, laps):
        base, slope = self.coefs[compound]
        a = np.asarray(ages, dtype=float)
        return (base + slope * a + self.hinges[compound] * np.maximum(0.0, a - self.knots[compound])
                + self.fuel_slope * np.asarray(laps, dtype=float))
