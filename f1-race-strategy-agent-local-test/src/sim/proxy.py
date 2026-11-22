"""Proxy pricing for the dry compounds a simulator's race laps do not model.

The full-window fit (the world) models a compound only when it has min_samples clean race
laps, and the to_snapshot fit (the belief) only the practice prior's compounds plus those with
min_samples clean race laps up to the snapshot. A compound nobody raced is then "Unknown" to
the world, so every plan using it is rejected. With SimConfig.compound_proxy.enabled (the
default) Simulator._fit runs `apply` on every fitted degradation model, so HARD, MEDIUM and
SOFT are all priced. A compound with race laps keeps its fit (tier 1); for the others each
parameter is a precision-weighted blend of every source below that has an estimate:

  tier 1  race laps: the fit itself, unchanged, for a compound with min_samples clean race
          rows in the fit (the belief: rows up to its calibration point).
  practice (tier 2)  practice long runs this weekend (FP sessions plus the Sprint, calibrate.practice_sessions):
          the practice fit's step from the anchor, base_c = base_a + (p_c - p_a) and
          wear_c = wear_a + (pw_c - pw_a). Additive in wear as well: an error in the fixed
          practice fuel prior biases every compound's practice wear by the same amount, which
          the difference cancels (a ratio would not, and a practice wear can be negative).
  qualifying (tier 3)  qualifying this weekend, pace only (qualifying_gaps): the compound gap in a
          fixed-effects fit of each driver's best timed lap per (segment, compound),
          lap = driver + segment + compound, slow laps trimmed as data.robust_filter trims
          them. base_c = base_a + (q_c - q_a). Fuel (every qualifying lap is on low fuel),
          the driver and car, track evolution and engine modes between segments (the segment
          term, which also takes the pre-2020 Q3 modes) cancel in the gap; what does not cancel
          is push against managed driving, which the validation measures as a bias.
  same_circuit (tier 4)  the same circuit in other seasons: not built. `_tier4_pace` and `_tier4_wear` are the
          extension point and return None.
  cross_race_step (tier 5)  the cross-race neighbour step (SimConfig.compound_proxy.steps, from
          configs/sim/compound_steps.json, built by compound_steps.py on calibration races
          only): base_c = base_a * (1 + pace_frac) per neighbour step, wear_c = wear_a *
          wear_ratio per step (inverted when pricing a harder compound from a softer one).
          Always present.

The blend (_blend): value = sum_i w_i v_i / sum_i w_i, each source's v_i from its own nearest
anchor, with fixed weights, one set for all races (SimConfig.compound_proxy.weights, from
configs/sim/compound_steps.json): w = 1 / the source's mean squared error in the
hidden-compound validation, fresh-pace error for the pace weights, wear error for the wear
weights (qualifying informs pace only). compound_steps.py computes them and validates the
blend leave-one-race-out (weights and step table from the other races).

Anchoring: every borrowed quantity is a step from an anchor compound a with tier-1 data in
this fit (the nearest one in SOFT < MEDIUM < HARD, ties to the one with more race rows), so
the race sets the level. A fit with no tier-1 compound at all (the belief before its first
race laps, the pre_race window, the practice fit) keeps its own compounds as they are (their
tier is 2, practice) and anchors the missing ones on them. A practice estimate needs the anchor
in the practice fit too, a qualifying one in qualifying; each source takes the nearest anchor
it can use.

Ordering: a softer compound is never slower fresh and never wears less than a harder one.
The constraint moves proxied values only, never a tier-1 one: a proxied base is clipped into
[max base of the softer compounds, min base of the harder ones] and a proxied wear into
[max wear of the harder ones, min wear of the softer ones], against the compounds priced so
far (tier 1 first, then the proxied compounds by their best source's tier, softest first). When tier-1 values
are themselves out of order the interval is empty and the value is clipped between them
("conflict"). A proxied wear is then floored at compound_proxy.min_wear_s_per_lap (0: a tyre
does not get faster with age) and at the fit's wear_lower_bound when it has one.

Uncertainty: a blended parameter is p = sum_a g_a * anchor_a + const, so fit_report["cov"]
gains its row as T cov T' + diag(tau^2), T the map from the fit's parameters: it moves with
its anchors in a posterior draw, plus its own tau. tau is the larger of compound_proxy.sd_floor
(the blend's leave-one-race-out error) and the cross-race step's spread across races times
the anchor value. The shrinkage weight of a proxied parameter is 0 (model_report source
"prior").

fit_report gains "proxy": {compound: {pace_tier (the best source's), pace_source (the sources
joined by "+"), pace_anchor, pace_sources ([{source, weight, anchor, value, step}]), the same
for wear, ordering, the steps and sds, qualifying_gap_s ({anchor, gap_s, sd_s, drivers}: the
measured qualifying gap to the nearest other raced compound, None without one)}} for every
one of HARD, MEDIUM and SOFT the model prices (raced ones included, source "race"), and the proxied compounds get the per compound
entries every model has (n_rows: practice rows behind it; n_race_rows_per_compound: the race
rows of the fit, 0 for a compound the fit left out; max_tyre_age: the practice fit's support,
NaN without one; shrinkage_weight 0; sd from the new cov; hinge dropped) and leave
"unmodelled".

`unraced_limits` gives R8 (rules.stint_limit) a limit for a compound no car ran in the race:
the oldest tyre age run on it in practice (the Sprint included on a sprint weekend), else the
nearest raced compound's limit times the tier-5 life ratio.
"""
import copy

import numpy as np

SOFTNESS = ("SOFT", "MEDIUM", "HARD")          # softest first
PAIRS = (("SOFT", "MEDIUM"), ("MEDIUM", "HARD"))
SOURCES = {1: "race", 2: "practice", 3: "qualifying", 4: "same_circuit", 5: "cross_race_step"}
# qualifying_gaps: the robust trim of data.robust_filter (k 2.5 robust sd, 2 passes, MAD floor)
QUALI_K_RESID, QUALI_N_ITER, QUALI_MIN_SCALE_S = 2.5, 2, 0.05
QUALI_MIN_DRIVERS = 2   # drivers linking a compound to another one before its gap is used


def pair_key(soft, hard) -> str:
    return f"{soft}-{hard}"


# --- the tier-5 table as of a race (docs/eval.md "Information contract") ---------------------
# A policy pricing race R reads the step table, the blend weights and the sd floor computed from
# the calibration races strictly before R (race_ids.as_of) only: compound_steps.json stores the
# per-race step rows ("per_race") and the forward-chained validation rows ("validation_rows":
# each race's hidden-compound errors priced with the table of the races before it), and
# as_of_config recomputes the three from them. With no earlier race the step is unavailable,
# every source weighs the same, and the sd floor is PRIOR_SD_FLOOR. The world (the judge) keeps
# the whole-set table.
STEP_KEYS = ("pace_frac", "wear_ratio", "life_ratio")
MIN_PAIR_ROWS = 50       # clean race rows on both compounds of a pair before a race's step counts
# a priori, not fitted: the blend's prior sd where no earlier race validates it, wider than the
# whole-set validation errors (pace 0.59 s, wear 0.079 s/lap)
PRIOR_SD_FLOOR = {"pace": 1.0, "wear": 0.1}
MIN_VALIDATION_ROWS = 3  # a source's weight (or the floor) needs this many earlier validation rows
WEIGHT_ERRORS = {"pace": "pace_err_s", "wear": "wear_err_s_per_lap"}
WEIGHT_SOURCES = {"pace": ("practice", "qualifying", "cross_race_step"), "wear": ("practice", "cross_race_step")}


def _finite(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def aggregate_steps(rows, exclude=(), keep_thin=False) -> dict:
    """{pair: {key: {value, sd, n, races}}} over per-race step rows (dicts or a frame's records):
    the median, the sample sd across races (NaN with one) and the count of the rows of races not
    in `exclude` with a finite value; a thin row (fewer than MIN_PAIR_ROWS clean race rows on
    either compound) is left out unless `keep_thin`."""
    rows = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
    rows = [r for r in rows if r["race_id"] not in set(exclude) and (keep_thin or not r.get("thin"))]
    out = {}
    for pair in sorted({r["pair"] for r in rows}):
        entry = {}
        for key in STEP_KEYS:
            v = sorted((r["race_id"], float(r[key])) for r in rows if r["pair"] == pair and _finite(r.get(key)))
            if not v:
                continue
            x = np.array([b for _, b in v])
            entry[key] = {"value": float(np.median(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                          "n": int(len(x)), "races": sorted(r for r, _ in v)}
        out[pair] = entry
    return out


def validation_weights(rows) -> dict:
    """{kind: {source: 1 / mean squared error}} of the single-source validation rows (case = the
    source) with at least MIN_VALIDATION_ROWS errors; a source without enough is left out, so
    the blend gives it 1.0 (proxy._blend), as every source has with no history at all."""
    out = {"pace": {}, "wear": {}}
    for kind, col in WEIGHT_ERRORS.items():
        for source in WEIGHT_SOURCES[kind]:
            e = [float(r[col]) for r in rows if r.get("case") == source and r.get("available") and _finite(r.get(col))]
            if len(e) >= MIN_VALIDATION_ROWS:
                out[kind][source] = 1.0 / float(np.mean(np.square(e)))
    return out


def validation_sd_floor(rows) -> dict:
    """{kind: the blend's root mean squared error} over the blend validation rows, PRIOR_SD_FLOOR
    for a kind with fewer than MIN_VALIDATION_ROWS of them."""
    out = {}
    for kind, col in WEIGHT_ERRORS.items():
        e = [float(r[col]) for r in rows if r.get("case") == "blend" and r.get("available") and _finite(r.get(col))]
        out[kind] = float(np.sqrt(np.mean(np.square(e)))) if len(e) >= MIN_VALIDATION_ROWS else PRIOR_SD_FLOOR[kind]
    return out


def history_races(cfg) -> list[str]:
    """Every race the as-of tier-5 inputs record (the step rows and the validation rows)."""
    return sorted({r["race_id"] for r in (cfg.per_race or [])} | {r["race_id"] for r in (cfg.validation or [])})


_AS_OF: dict = {}


def as_of_config(cfg, race_id):
    """`cfg` (SimConfig.compound_proxy) with steps, weights and sd_floor recomputed from the
    races strictly before `race_id` (race_ids.as_of) in its per_race and validation rows; a race
    id with no date (a synthetic "syn", say) has no earlier race. Cached per set of races."""
    from .race_ids import as_of, canonical_race_id
    try:
        earlier = tuple(as_of(canonical_race_id(race_id), history_races(cfg)))
    except ValueError:
        earlier = ()
    key = (repr(cfg.per_race), repr(cfg.validation), earlier)
    if key not in _AS_OF:
        keep = set(earlier)
        val = [r for r in (cfg.validation or []) if r["race_id"] in keep]
        _AS_OF[key] = {"steps": aggregate_steps([r for r in (cfg.per_race or []) if r["race_id"] in keep]),
                       "weights": validation_weights(val), "sd_floor": validation_sd_floor(val),
                       "races": list(earlier)}
    got = _AS_OF[key]
    return cfg.model_copy(update={k: got[k] for k in ("steps", "weights", "sd_floor")})


def _idx(c):
    return SOFTNESS.index(c)


# --- tier 3: qualifying ---------------------------------------------------------------------
def _design(g, comps, ref):
    """Columns of lap = driver + segment + compound on the group bests `g`: one per driver, one
    per segment after the first, one per compound other than `ref`. Returns (X, names)."""
    cols, names = [], []
    for d in sorted(g["driver"].unique()):
        cols.append((g["driver"] == d).to_numpy(dtype=float))
        names.append(("driver", d))
    for s in sorted(g["segment"].unique())[1:]:
        cols.append((g["segment"] == s).to_numpy(dtype=float))
        names.append(("segment", s))
    for c in comps:
        if c != ref:
            cols.append((g["compound"] == c).to_numpy(dtype=float))
            names.append(("compound", c))
    return np.column_stack(cols), names


def _linking(g, c):
    """Drivers with a group on `c` and on another compound."""
    on = set(g.loc[g["compound"] == c, "driver"])
    other = set(g.loc[g["compound"] != c, "driver"])
    return on & other


def _identified(g):
    """The compounds of `g` with QUALI_MIN_DRIVERS linking drivers, softest first, and the
    reference: the one with the most rows (ties: the softer)."""
    comps = sorted((c for c in g["compound"].unique() if len(_linking(g, c)) >= QUALI_MIN_DRIVERS), key=_idx)
    ref = max(comps, key=lambda c: ((g["compound"] == c).sum(), -_idx(c))) if comps else None
    return comps, ref


def qualifying_gaps(q):
    """The tier-3 compound gaps from a data.qualifying_frame, or None.

    Each driver's best timed lap per (segment, compound) is one row; the model is
    lap = driver + segment + compound by least squares. As data.robust_filter does, it is fitted
    QUALI_N_ITER + 1 times, each time on the rows whose residual against the previous fit is
    below QUALI_K_RESID robust sds (scaled MAD over every row, floored at QUALI_MIN_SCALE_S;
    fast rows never cut), then once more on the rows kept. A compound enters only with
    QUALI_MIN_DRIVERS drivers who also have a row on another compound (otherwise its gap is the
    driver term's). Returns {"gap": {compound: s against the reference compound}, "cov":
    {(c, d): covariance}, "n_drivers": {compound: linking drivers}, "n_rows": rows kept,
    "reference": compound}; None when fewer than two compounds are identified."""
    if q is None or len(q) == 0:
        return None
    t = q[q["timed"].astype(bool) & q["compound"].isin(SOFTNESS)]
    t = t.assign(segment=t["segment"].fillna("Q"))
    g = t.groupby(["driver", "segment", "compound"], as_index=False)["lap_time_s"].min()
    comps, ref = _identified(g)
    if len(comps) < 2:
        return None
    d = g[g["compound"].isin(comps)].reset_index(drop=True)
    X, names = _design(d, comps, ref)
    y = d["lap_time_s"].to_numpy(dtype=float)
    keep = np.ones(len(d), dtype=bool)
    for _ in range(QUALI_N_ITER + 1):
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        resid = y - X @ beta
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), QUALI_MIN_SCALE_S)
        keep = resid < QUALI_K_RESID * scale
    k = d[keep]
    comps, ref = _identified(k)
    if len(comps) < 2:
        return None
    k = k[k["compound"].isin(comps)]
    X, names = _design(k, comps, ref)
    y = k["lap_time_s"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n = X.shape[0]
    rank = np.linalg.matrix_rank(X)
    s2 = float(((y - X @ beta) ** 2).sum() / (n - rank)) if n > rank else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    at = {nm: j for j, nm in enumerate(names)}
    gap = {ref: 0.0}
    for c in comps:
        if c != ref:
            gap[c] = float(beta[at[("compound", c)]])

    def v(a, b):
        if a == ref or b == ref:
            return 0.0
        return float(cov[at[("compound", a)], at[("compound", b)]])
    return {"gap": gap, "cov": {(a, b): v(a, b) for a in comps for b in comps},
            "n_drivers": {c: len(_linking(k, c)) for c in comps}, "n_rows": int(len(k)), "reference": ref}


def quali_step(gaps, c, a):
    """(q_c - q_a, its sd, linking drivers on c) or None when either compound is not in `gaps`."""
    if gaps is None or c not in gaps["gap"] or a not in gaps["gap"]:
        return None
    cv = gaps["cov"]
    var = cv[(c, c)] + cv[(a, a)] - 2 * cv[(c, a)]
    return gaps["gap"][c] - gaps["gap"][a], float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else float("nan"), \
        gaps["n_drivers"][c]


# --- tier 5: the cross-race neighbour step ---------------------------------------------------
def neighbour_factor(steps, a, c, key):
    """(factor g, relative sd) that takes anchor a's value to compound c's along the neighbour
    steps for `key` (pace_frac: g = prod (1 + f) towards the softer end, its inverse towards
    the harder; wear_ratio and life_ratio: prod r, or 1 / r). None when a step is missing or
    degenerate. The relative sd is sqrt(sum (sd / |step factor|)^2), the delta method."""
    i, j = _idx(a), _idx(c)
    g, rel2 = 1.0, 0.0
    lo, hi = sorted((i, j))
    for k in range(lo, hi):
        entry = (steps or {}).get(pair_key(SOFTNESS[k], SOFTNESS[k + 1]), {}).get(key)
        if entry is None:
            return None
        f = 1.0 + entry["value"] if key == "pace_frac" else entry["value"]
        if not np.isfinite(f) or f == 0:
            return None
        rel2 += (entry["sd"] / abs(f)) ** 2
        g *= f if j < i else 1.0 / f   # towards the softer end: times the factor
    return g, float(np.sqrt(rel2))


# --- extension point: tier 4 (same circuit, other seasons) -----------------------------------
def _tier4_pace(c, a):
    return None


def _tier4_wear(c, a):
    return None


# --- apply -----------------------------------------------------------------------------------
def _anchors_by_distance(c, anchors, race_rows):
    return sorted(anchors, key=lambda a: (abs(_idx(a) - _idx(c)), -race_rows.get(a, 0), _idx(a)))


def _cov_of(model):
    cov = (model.fit_report or {}).get("cov") if model is not None else None
    if not cov:
        return None
    return {n: i for i, n in enumerate(cov["names"])}, np.array(cov["matrix"], dtype=float)


def _diff_sd(model, c, a, param):
    got = _cov_of(model)
    if got is None:
        return float("nan")
    at, m = got
    i, j = at.get(f"{param}[{c}]"), at.get(f"{param}[{a}]")
    if i is None or j is None:
        return float("nan")
    var = m[i, i] + m[j, j] - 2 * m[i, j]
    return float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else float("nan")


SOURCE_TIER = {"practice": 2, "qualifying": 3, "same_circuit": 4, "cross_race_step": 5}
SOURCES_ALL = ("practice", "qualifying", "cross_race_step")
SOURCE_WORDS = {"race": "race laps", "practice": "practice long runs", "qualifying": "qualifying",
                "same_circuit": "the same circuit in other seasons", "cross_race_step": "the cross-race step"}


def _part(kind, source, c, a, coefs, practice, quali, steps):
    """One source's estimate of c's base (`kind` "pace") or wear ("wear") from anchor a, as
    {source, anchor, g, value, sd, step} with value = g * anchor's value + a step; None when the
    source has nothing for (c, a)."""
    j = 0 if kind == "pace" else 1
    pc = practice.coefs if practice is not None else {}
    if source == "practice" and c in pc and a in pc:
        d = pc[c][j] - pc[a][j]
        sd = _diff_sd(practice, c, a, "base_s" if kind == "pace" else "wear_s_per_lap")
        return {"source": source, "anchor": a, "g": 1.0, "value": coefs[a][j] + d, "sd": sd, "step": d}
    if source == "qualifying" and kind == "pace":
        step = quali_step(quali, c, a)
        if step is not None:
            d, sd, n = step
            return {"source": source, "anchor": a, "g": 1.0, "value": coefs[a][j] + d, "sd": sd, "step": d,
                    "drivers": n}
    if source == "same_circuit" and (_tier4_pace if kind == "pace" else _tier4_wear)(c, a) is not None:  # pragma: no cover
        raise NotImplementedError("tier 4 (same circuit, other seasons) is not built")
    if source == "cross_race_step":
        f = neighbour_factor(steps, a, c, "pace_frac" if kind == "pace" else "wear_ratio")
        if f is not None:
            g, rel = f
            value = coefs[a][j] * g
            return {"source": source, "anchor": a, "g": g, "value": value, "sd": abs(value) * rel,
                    "step": value - coefs[a][j]}
    return None


def _blend(kind, c, anchors, coefs, practice, quali, cfg, race_rows, sources):
    """The precision-weighted blend of every source in `sources` with an estimate of c's base
    (`kind` "pace") or wear ("wear"), each from its nearest anchor; None when none has one.
    Returns {value, coef: {anchor: g_eff}, tau, tier, source, parts}: value = sum_i w_i v_i /
    sum_i w_i with the fixed weights cfg.weights[kind][source] (1 / the source's mean squared
    error in the hidden-compound validation; 1.0 for a source without a weight), linear in the
    anchors with g_eff = sum_i (w_i / W) g_i. tau, the blend's prior sd, is the larger of
    cfg.sd_floor[kind] (the blend's leave-one-race-out error) and the cross-race step's spread."""
    order = _anchors_by_distance(c, anchors, race_rows)
    parts = []
    for source in sources:
        for a in order:
            part = _part(kind, source, c, a, coefs, practice, quali, cfg.steps)
            if part is not None:
                part["weight"] = float(((cfg.weights or {}).get(kind) or {}).get(source, 1.0))
                parts.append(part)
                break
    if not parts:
        return None
    total = sum(p["weight"] for p in parts)
    if total <= 0:
        for p in parts:
            p["weight"] = 1.0
        total = float(len(parts))
    coef = {}
    for p in parts:
        p["weight"] = p["weight"] / total
        coef[p["anchor"]] = coef.get(p["anchor"], 0.0) + p["weight"] * p["g"]
    value = sum(p["weight"] * p["value"] for p in parts)
    spread = max([p["sd"] for p in parts if p["source"] == "cross_race_step" and np.isfinite(p["sd"])],
                 default=0.0)
    tau = max(float((cfg.sd_floor or {}).get(kind, 0.0)), spread)
    if tau <= 0:   # no floor and no step: the sources' own sd
        tau = max([p["sd"] for p in parts if np.isfinite(p["sd"])], default=0.0)
    return {"value": float(value), "coef": coef, "tau": float(tau),
            "tier": min(SOURCE_TIER[p["source"]] for p in parts),
            "source": "+".join(p["source"] for p in parts), "parts": parts}


def _main_anchor(blend):
    """The anchor carrying the most weight in a blend."""
    return max(blend["coef"], key=lambda a: (blend["coef"][a], -_idx(a)))


def _parts_report(blend):
    out = []
    for p in blend["parts"]:
        e = {"source": p["source"], "weight": float(p["weight"]), "anchor": p["anchor"], "value": float(p["value"]),
             "step": float(p["step"])}
        if "drivers" in p:
            e["drivers"] = int(p["drivers"])
        out.append(e)
    return out


def _qualifying_gaps(info, anchors, quali, race_rows):
    """For every compound in `info`, its measured qualifying gap to the nearest other anchor
    (quali_step), the fresh-tyre datapoint the agent's model report shows: {anchor, gap_s, sd_s,
    drivers}, None when qualifying has nothing for it."""
    for c in info:
        gap = None
        for a in _anchors_by_distance(c, [x for x in anchors if x != c], race_rows):
            step = quali_step(quali, c, a)
            if step is not None:
                d, sd, n = step
                gap = {"anchor": a, "gap_s": float(d), "sd_s": float(sd), "drivers": int(n)}
                break
        info[c]["qualifying_gap_s"] = gap


def _clip(value, lo, hi):
    """(value clipped into [lo, hi], note); an empty interval (lo > hi) clips into [hi, lo]."""
    if lo is None and hi is None:
        return value, None
    if lo is not None and hi is not None and lo > hi:
        return min(max(value, hi), lo), "conflict"
    if lo is not None and value < lo:
        return lo, "raised"
    if hi is not None and value > hi:
        return hi, "lowered"
    return value, None


def _transform(M, T):
    """T M T', NaN-safe: an entry only reads the block of M its rows of T touch."""
    n = T.shape[0]
    out = np.zeros((n, n))
    nz = [np.flatnonzero(T[i]) for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            if len(nz[i]) and len(nz[j]):
                v = T[i, nz[i]] @ M[np.ix_(nz[i], nz[j])] @ T[j, nz[j]]
            else:
                v = 0.0
            out[i, j] = out[j, i] = v
    return out


def apply(model, cfg, *, min_samples, practice=None, quali=None, knot=None, wear_lower_bound=None,
          sources=None):
    """`model` with HARD, MEDIUM and SOFT priced by the blend (module docstring), a copy;
    `model` itself when it prices none of them or has no fit_report. `cfg` is
    SimConfig.compound_proxy, `practice` the practice fit the tier-2 steps come from (the belief's
    prior, else Simulator.physics_prior), `quali` qualifying_gaps' result, `knot(c)` the hinge knot
    of a new compound on a hinge model, `wear_lower_bound` the fit's wear bound (None: none).
    `sources` (default cfg.sources; a list, or {"pace": [...], "wear": [...]}) are the sources
    the blend may use; the validation (compound_steps.py) prices through one source at a time."""
    sources = cfg.sources if sources is None else sources
    sources = ({"pace": list(sources["pace"]), "wear": list(sources["wear"])} if isinstance(sources, dict)
               else {"pace": list(sources), "wear": list(sources)})
    rep0 = getattr(model, "fit_report", None)
    if rep0 is None or "cov" not in rep0:
        return model
    coefs = dict(model.coefs)
    race_rows = dict(rep0.get("n_race_rows_per_compound") or {})
    present = [c for c in SOFTNESS if c in coefs]
    if not present:
        return model
    tier1 = [c for c in present if race_rows.get(c, 0) >= min_samples]
    anchors = tier1 or present
    targets = [c for c in SOFTNESS if c not in anchors]
    pcoefs = practice.coefs if practice is not None else {}
    info = {}

    def as_fitted(c):
        """The label of a compound the fit prices itself: race laps, or practice (the prior)
        for one with fewer than min_samples race rows."""
        tier, src = (1, SOURCES[1]) if c in tier1 or c not in pcoefs else (2, SOURCES[2])
        return {"pace_tier": tier, "pace_source": src, "pace_anchor": None, "pace_sources": [],
                "wear_tier": tier, "wear_source": src, "wear_anchor": None, "wear_sources": [], "ordering": []}
    for c in anchors:
        info[c] = as_fitted(c)
    priced = {}
    for c in targets:
        pace = _blend("pace", c, anchors, coefs, practice, quali, cfg, race_rows, sources["pace"])
        wear = _blend("wear", c, anchors, coefs, practice, quali, cfg, race_rows, sources["wear"])
        if pace is None or wear is None:
            if c in coefs:   # a compound of the fit no source can re-price: kept as the fit has it
                info[c] = as_fitted(c)
            continue
        priced[c] = (pace, wear)
    if not priced:
        _qualifying_gaps(info, anchors, quali, race_rows)
        out = copy.copy(model)
        out.fit_report = {**rep0, "proxy": {c: info[c] for c in SOFTNESS if c in info}}
        return out
    # ordering: settle the anchors (and compounds kept as fitted), then the proxied compounds by
    # pace tier, softest first
    values = {c: list(coefs[c]) for c in coefs if c in SOFTNESS and c not in priced}
    notes = {}
    for c in sorted(priced, key=lambda c: (priced[c][0]["tier"], _idx(c))):
        base, wear = priced[c][0]["value"], priced[c][1]["value"]
        softer = [values[s] for s in values if _idx(s) < _idx(c)]
        harder = [values[h] for h in values if _idx(h) > _idx(c)]
        note = []
        base, why = _clip(base, max((v[0] for v in softer), default=None), min((v[0] for v in harder), default=None))
        if why:
            note.append(f"base {why}")
        wear, why = _clip(wear, max((v[1] for v in harder), default=None), min((v[1] for v in softer), default=None))
        if why:
            note.append(f"wear {why}")
        floor = max([cfg.min_wear_s_per_lap] + ([wear_lower_bound] if wear_lower_bound is not None else []))
        if wear < floor:
            wear = float(floor)
            note.append("wear floored")
        values[c] = [float(base), float(wear)]
        notes[c] = note
    # the new parameter vector and its covariance
    old_cov = rep0["cov"]
    old_names = list(old_cov["names"])
    M = np.array(old_cov["matrix"], dtype=float).reshape(len(old_names), len(old_names))
    at = {n: i for i, n in enumerate(old_names)}
    comps = sorted(set(coefs) | set(priced))
    new_names = [f"{p}[{c}]" for c in comps for p in ("base_s", "wear_s_per_lap")]
    new_names += [n for n in old_names if not n.startswith(("base_s[", "wear_s_per_lap["))]
    T = np.zeros((len(new_names), len(old_names)))
    tau2 = np.zeros(len(new_names))
    for i, n in enumerate(new_names):
        p, _, c = n[:-1].partition("[") if n.endswith("]") else (n, None, None)
        if c in priced:
            blend = priced[c][0 if p == "base_s" else 1]
            for a, g in blend["coef"].items():
                T[i, at[f"{p}[{a}]"]] += g
            tau2[i] = blend["tau"] ** 2
        else:
            T[i, at[n]] = 1.0
    M_new = _transform(M, T) + np.diag(tau2)
    new_coefs = {c: (tuple(values[c]) if c in priced else coefs[c]) for c in comps}
    rep = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rep0.items()}
    for key in ("n_rows", "n_race_rows_per_compound", "max_tyre_age", "unmodelled"):
        rep[key] = dict(rep0.get(key) or {})
    rep["shrinkage_weight"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rep0["shrinkage_weight"].items()}
    rep["sd"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rep0["sd"].items()}
    rep["cov"] = {"names": new_names, "matrix": [[float(x) for x in row] for row in M_new]}
    psupport = (practice.fit_report or {}) if practice is not None else {}
    new_at = {n: i for i, n in enumerate(new_names)}
    for c in priced:
        rep["n_rows"].setdefault(c, int((psupport.get("n_rows") or {}).get(c, 0)))
        rep["n_race_rows_per_compound"].setdefault(c, 0)
        rep["max_tyre_age"].setdefault(c, float((psupport.get("max_tyre_age") or {}).get(c, float("nan"))))
        rep["shrinkage_weight"]["base_s"][c] = 0.0
        rep["shrinkage_weight"]["wear_s_per_lap"][c] = 0.0
        rep["sd"]["base_s"][c] = float(np.sqrt(max(M_new[new_at[f"base_s[{c}]"]] [new_at[f"base_s[{c}]"]], 0.0)))
        rep["sd"]["wear_s_per_lap"][c] = float(np.sqrt(max(M_new[new_at[f"wear_s_per_lap[{c}]"]]
                                                           [new_at[f"wear_s_per_lap[{c}]"]], 0.0)))
        rep["unmodelled"].pop(c, None)
        pb, wb = priced[c]
        pa, wa = _main_anchor(pb), _main_anchor(wb)
        info[c] = {"pace_tier": pb["tier"], "pace_source": pb["source"], "pace_anchor": pa,
                   "pace_sources": _parts_report(pb), "wear_tier": wb["tier"], "wear_source": wb["source"],
                   "wear_anchor": wa, "wear_sources": _parts_report(wb), "ordering": notes[c],
                   "pace_step_s": float(values[c][0] - coefs[pa][0]),
                   "wear_step_s_per_lap": float(values[c][1] - coefs[wa][1]),
                   "pace_sd_s": float(pb["tau"]), "wear_sd_s_per_lap": float(wb["tau"])}
    if "bounded" in rep:
        rep["bounded"] = dict(rep0["bounded"])
        for c in priced:
            rep["bounded"][c] = bool(wear_lower_bound is not None and values[c][1] <= wear_lower_bound)
    unc = rep0.get("unconstrained")
    if unc is not None:
        um = np.array(unc["matrix"], dtype=float).reshape(len(old_names), len(old_names))
        entry = {"names": new_names, "matrix": [[float(x) for x in row] for row in _transform(um, T) + np.diag(tau2)]}
        point_old = np.array([_value(model, n) for n in old_names])
        point_new = np.array([_value_of(new_coefs, model, n) for n in new_names])
        const = point_new - T @ point_old
        if "mean" in unc:
            entry["mean"] = (T @ np.asarray(unc["mean"], dtype=float) + const).tolist()
        if "shift" in unc:
            entry["shift"] = (T @ np.asarray(unc["shift"], dtype=float)).tolist()
        rep["unconstrained"] = entry
    out = copy.copy(model)
    out.__dict__.pop("_draw_plan", None)
    out.coefs = new_coefs
    hinges = getattr(model, "hinges", None)
    if hinges is not None:
        from .degradation import _dropped
        out.hinges = {**hinges, **{c: 0.0 for c in priced if c not in hinges}}
        out.knots = {**model.knots, **{c: int(knot(c)) for c in priced if c not in model.knots}}
        rep["hinge"] = {**rep0.get("hinge", {}), **{c: _dropped(out.knots[c], 0) for c in priced
                                                    if c not in (rep0.get("hinge") or {})}}
    _qualifying_gaps(info, anchors, quali, race_rows)
    rep["proxy"] = {c: info[c] for c in SOFTNESS if c in info}
    out.fit_report = rep
    return out


def _value(model, name):
    p, _, c = name[:-1].partition("[") if name.endswith("]") else (name, None, None)
    if p == "fuel_s_per_lap":
        return float(model.fuel_slope)
    if p == "hinge_s_per_lap":
        return float(model.hinges[c])
    return float(model.coefs[c][0 if p == "base_s" else 1])


def _value_of(coefs, model, name):
    p, _, c = name[:-1].partition("[") if name.endswith("]") else (name, None, None)
    if p in ("base_s", "wear_s_per_lap"):
        return float(coefs[c][0 if p == "base_s" else 1])
    return _value(model, name)


def proxied(fit_report) -> dict:
    """{compound: {"pace": source, "wear": source}} for the compounds the fit prices by proxy
    (a pace or wear tier other than 1); {} without proxying."""
    out = {}
    for c, e in ((fit_report or {}).get("proxy") or {}).items():
        if e["pace_tier"] != 1 or e["wear_tier"] != 1:
            out[c] = {"pace": e["pace_source"], "wear": e["wear_source"]}
    return out


def describe(compound, entry) -> str:
    """'SOFT: pace from practice long runs 40%, qualifying 35%, the cross-race step 25% (anchor
    MEDIUM); wear from practice long runs 30%, the cross-race step 70% (anchor MEDIUM)', or
    'MEDIUM: pace from race laps; wear from race laps'; a measured qualifying gap is appended."""
    def part(kind):
        parts = entry.get(f"{kind}_sources") or []
        if not parts:
            src = entry[f"{kind}_source"]
            return f"{kind} from {SOURCE_WORDS.get(src, src)}"
        what = ", ".join(f"{SOURCE_WORDS[p['source']]} {100 * p['weight']:.0f}%" for p in parts)
        anchors = sorted({p["anchor"] for p in parts}, key=_idx)
        return f"{kind} from {what} (anchor {'/'.join(anchors)})"
    text = f"{compound}: {part('pace')}; {part('wear')}"
    gap = entry.get("qualifying_gap_s")
    if gap:
        text += (f"; qualifying gap to {gap['anchor']} {gap['gap_s']:+.2f} s on fresh tyres "
                 f"({gap['drivers']} drivers)")
    return text


# --- R8 for an unraced compound --------------------------------------------------------------
def unraced_limits(limits, practice, steps):
    """(limits, sources): rules.stint_limits plus, for each of HARD, MEDIUM and SOFT no car ran
    in the race, the oldest tyre age run on it in `practice` (every practice row, the Sprint
    included), else the nearest raced compound's limit times the tier-5 life ratio (rounded
    down to a whole lap). sources names where each added limit came from: "practice", or
    "life_ratio:<anchor>". Nothing is added when no dry compound was raced."""
    limits, sources = dict(limits), {}
    raced = [c for c in SOFTNESS if c in limits]
    if not raced:
        return limits, sources
    ages = {}
    if practice is not None and len(practice) and "tyre_age" in practice:
        p = practice[practice["compound"].isin(SOFTNESS) & practice["tyre_age"].notna()]
        ages = {str(c): float(a) for c, a in p.groupby("compound")["tyre_age"].max().items()}
    for c in SOFTNESS:
        if c in limits:
            continue
        if c in ages:
            limits[c], sources[c] = ages[c], "practice"
            continue
        for a in sorted(raced, key=lambda a: (abs(_idx(a) - _idx(c)), _idx(a))):
            f = neighbour_factor(steps, a, c, "life_ratio")
            if f is not None:
                limits[c], sources[c] = float(np.floor(limits[a] * f[0])), f"life_ratio:{a}"
                break
    return limits, sources
