"""The judge's thin-fit wear guard (SimConfig.wear_guard, off by default; docs/eval.md "Thin wear fits").

    python -m src.sim.wear_guard build      # data/derived/wear_history.csv from the judge's own fits
    python -m src.sim.wear_guard report     # the fits the guard changes on the calibration races

A compound's wear slope in the judge's whole-race fit can rest on a handful of laps: 2022 Brazil's
HARD wear is 0.160 s/lap from 10 clean rows of one driver at ages 2-11, 2023 Bahrain's MEDIUM
0.313 from 8 rows. With the guard on, a compound with fewer than `min_drivers` drivers or
`min_rows` clean race rows (and at least one: a compound with none is the proxy's, left alone)
has its wear shrunk toward the as-of historical wear by precision:

    w' = (w / sd^2 + p / sd_p^2) / (1 / sd^2 + 1 / sd_p^2)

sd the fit's own standard error of w (fit_report), p and sd_p the historical value (wear_prior).
The base moves by (w - w') x the mean tyre age of the compound's fit rows, so the fit's mean lap
on that compound is kept. Fuel and the other compounds are untouched.

wear_prior(race_id, compound) reads data/derived/wear_history.csv, the judge's own fits (the
default world, guard off) of every cached non-held-out race. It takes only races strictly before
`race_id` (race_ids.as_of), only well-supported fits (WELL_DRIVERS drivers and WELL_ROWS rows), and
regresses wear on the compound's place on the common hardness scale (the nomination's common
index, known before the race); with fewer than MIN_HISTORY such fits, the a priori ROLE_PRIOR.
It never reads a held-out race (build refuses them). Registered in provenance ("wear_history").
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .race_ids import as_of, canonical_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
TABLE = ROOT / "data" / "derived" / "wear_history.csv"
MANIFEST = ROOT / "data" / "derived" / "wear_history_manifest.json"
EVAL_SET = ROOT / "data" / "eval_set.json"
NEVER = frozenset({"2019_06_monaco", "2020_16_sakhir", "2025_11_austria", "2024_11_austria"})
DRY = ("SOFT", "MEDIUM", "HARD")
WELL_DRIVERS, WELL_ROWS = 5, 60   # a history fit counts when it rests on this much
MIN_HISTORY = 8                   # fits needed for the regression on the hardness index
SD_FLOOR = 0.02                   # s/lap: the prior's sd is at least this
# a priori, fitted on no race: wear by role (s/lap) and its sd, when the history is too thin
ROLE_PRIOR = {"HARD": 0.04, "MEDIUM": 0.06, "SOFT": 0.08}
ROLE_PRIOR_SD = 0.05
# a race with no compound data (wet, red-flagged) or with a mandated stint cap
EXCLUDED = ("2025_23_qatar",)


def held_out() -> set[str]:
    races = json.loads(EVAL_SET.read_text()).get("races", []) if EVAL_SET.is_file() else []
    return {canonical_race_id(r) for r in races} | set(NEVER)


def fit_rows(frame) -> pd.DataFrame:
    """Clean race rows per compound: rows, drivers, mean tyre age."""
    f = frame[frame["lap"].notna()]
    if "clean" in f:
        f = f[f["clean"].astype(bool)]
    g = f.groupby("compound")
    return pd.DataFrame({"n_rows": g.size(), "n_drivers": g["driver"].nunique(),
                         "mean_age": g["tyre_age"].mean(), "age_max": g["tyre_age"].max()})


def _one(rid):
    from src.contracts import Snapshot
    from . import jobs
    from .compute import load_profile
    from .config import SimConfig, world_default
    from .engine import Simulator
    from .tyre_history import race_info
    data = jobs.load_race(rid, load_profile())
    sim = Simulator(data.frame, world_default(SimConfig()), data.practice)
    snap = Snapshot(race=data.race, driver=str(data.frame["driver"].iloc[0]), laps_completed=0, compound="MEDIUM",
                    tyre_age=0, elapsed_s=0.0)
    fit = sim.fitted_for(snap)
    rep = fit.degradation.fit_report or {}
    sd = rep.get("sd", {}).get("wear_s_per_lap", {})
    rows = fit_rows(data.frame)
    idx = race_info(rid).index
    out = []
    for c, (_, w) in fit.degradation.coefs.items():
        if c not in DRY or c not in rows.index:
            continue
        r = rows.loc[c]
        out.append({"race_id": rid, "compound": c, "common_index": idx.get(c, np.nan), "wear_s": float(w),
                    "wear_sd": float(sd.get(c, np.nan)), "n_rows": int(r["n_rows"]), "n_drivers": int(r["n_drivers"]),
                    "age_max": float(r["age_max"])})
    return out


def build(races=None, jobs_n=3, log=print) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    from .tyre_history import cached_races
    held = held_out()
    races = [canonical_race_id(r) for r in (races or [r for r, _ in cached_races() if r not in held])]
    bad = [r for r in races if r in held]
    if bad:
        raise ValueError(f"{bad} held out: never built into the wear history")
    rows = []
    with ProcessPoolExecutor(jobs_n) as ex:
        for rid, res in zip(races, ex.map(_safe, races)):
            log(f"  {rid}: {len(res)} compounds" if isinstance(res, list) else f"  {rid}: skipped ({res})")
            if isinstance(res, list):
                rows += res
    t = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLE, index=False, float_format="%.6g")
    MANIFEST.write_text(json.dumps({
        "command": "python -m src.sim.wear_guard build",
        "description": "The judge's own whole-race wear fit (default world, guard off) per race and dry compound; "
                       "read as of the race by wear_guard.wear_prior.",
        "races": sorted(t["race_id"].unique()), "held_out_never_loaded": sorted(held)}, indent=1) + "\n")
    return t


def _safe(rid):
    try:
        return _one(rid)
    except Exception as e:   # noqa: BLE001 - a race that does not load is reported, not fatal
        return f"{type(e).__name__}: {e}"


@lru_cache(maxsize=1)
def load_table() -> pd.DataFrame:
    t = pd.read_csv(TABLE)
    t["race_id"] = t["race_id"].map(canonical_race_id)
    return t


@lru_cache(maxsize=1)
def _wet_races() -> frozenset:
    """Races with wet running (tyre_history wet_share > 0): their dry fits mix a drying track in."""
    from .tyre_history import HISTORY
    if not Path(HISTORY).is_file():
        return frozenset()
    h = pd.read_csv(HISTORY)
    return frozenset(map(canonical_race_id, h.loc[pd.to_numeric(h["wet_share"], errors="coerce") > 0, "race_id"]))


def _well(t):
    return t[(t["n_drivers"] >= WELL_DRIVERS) & (t["n_rows"] >= WELL_ROWS) & ~t["race_id"].isin(EXCLUDED)
             & ~t["race_id"].isin(_wet_races()) & np.isfinite(t["wear_s"])]


def races_for(race_id) -> list[str]:
    """The races wear_prior reads for `race_id` (provenance)."""
    if not TABLE.is_file():
        return []
    t = load_table()
    return as_of(race_id, sorted(set(t["race_id"]) - held_out()))


def recorded() -> list[str]:
    return sorted(load_table()["race_id"].unique()) if TABLE.is_file() else []


@lru_cache(maxsize=512)
def wear_prior(race_id, compound) -> tuple[float, float, str]:
    """(wear s/lap, sd, source) as of `race_id` for `compound` (module docstring)."""
    try:
        earlier = set(races_for(canonical_race_id(race_id)))
    except (TypeError, ValueError):
        earlier = set()
    t = _well(load_table())[lambda x: x["race_id"].isin(earlier)] if TABLE.is_file() else pd.DataFrame()
    if len(t) >= MIN_HISTORY and compound in DRY:
        from .tyre_history import race_info
        try:
            x0 = race_info(race_id).index.get(compound, np.nan)
        except Exception:   # noqa: BLE001
            x0 = np.nan
        tt = t[np.isfinite(t["common_index"])]
        if np.isfinite(x0) and len(tt) >= MIN_HISTORY:
            X = np.column_stack([np.ones(len(tt)), tt["common_index"].to_numpy(float)])
            b = np.linalg.lstsq(X, tt["wear_s"].to_numpy(float), rcond=None)[0]
            res = tt["wear_s"].to_numpy(float) - X @ b
            return float(b[0] + b[1] * x0), max(SD_FLOOR, float(np.std(res, ddof=2))), f"history_index:{len(tt)}"
        same = t[t["compound"] == compound]["wear_s"]
        if len(same) >= 3:
            return float(same.median()), max(SD_FLOOR, float(same.std(ddof=1))), f"history_role:{len(same)}"
    return ROLE_PRIOR.get(compound, float(np.mean(list(ROLE_PRIOR.values())))), ROLE_PRIOR_SD, "a_priori"


def guarded(model, frame, race_id, guard) -> tuple[object, dict]:
    """A copy of `model` with thin compounds' wear shrunk (module docstring), and {compound: change}."""
    import copy
    rows = fit_rows(frame)
    rep = getattr(model, "fit_report", None) or {}
    sds = rep.get("sd", {}).get("wear_s_per_lap", {})
    coefs = dict(model.coefs)
    changes = {}
    for c, (base, w) in model.coefs.items():
        if c not in rows.index:
            continue   # no race rows: the proxy's compound, left alone
        r = rows.loc[c]
        if r["n_rows"] < 1 or (r["n_drivers"] >= guard.min_drivers and r["n_rows"] >= guard.min_rows):
            continue
        sd = float(sds.get(c, np.nan))
        if getattr(guard, "scale_sd_by_drivers", False) and r["n_drivers"] < guard.min_drivers:
            sd *= float(np.sqrt(guard.min_drivers / max(1, r["n_drivers"])))
        p, psd, src = wear_prior(race_id, c)
        if not np.isfinite(sd) or sd <= 0:
            w2 = p
        else:
            w2 = (w / sd ** 2 + p / psd ** 2) / (1 / sd ** 2 + 1 / psd ** 2)
        base2 = base + (w - w2) * float(r["mean_age"])
        coefs[c] = (base2, w2) if not isinstance(model.coefs[c], list) else [base2, w2]
        changes[c] = {"wear_s": float(w), "wear_sd": sd, "prior": p, "prior_sd": psd, "prior_source": src,
                      "wear_guarded_s": float(w2), "base_shift_s": float(base2 - base),
                      "n_rows": int(r["n_rows"]), "n_drivers": int(r["n_drivers"])}
    if not changes:
        return model, {}
    out = copy.copy(model)
    out.coefs = coefs
    if rep:
        out.fit_report = {**rep, "wear_guard": changes}
    return out, changes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.wear_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--races", nargs="*")
    b.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        t = build(a.races, a.jobs)
        print(t.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
