"""ModelReport and support warnings: what the fitted degradation model can and cannot say.

Both read a model's fit_report (degradation.py). Simulator.model_report and
Simulator.simulate_batch are the entry points.
"""
import math
import os

import pandas as pd

from ..contracts import CoefficientReport, CompoundReport, ModelReport, Plan, Snapshot

CALIBRATION_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration",
                                   "midrace_summary.csv")
WEAK_WEIGHT = 0.5    # a parameter whose data share of the posterior precision is below this is flagged
FITTED_WEIGHT = 0.9  # source "fitted" at or above this weight
PRIOR_WEIGHT = 0.1   # source "prior" at or below this weight, "shrunk" in between


def _source(weight):
    return "fitted" if weight >= FITTED_WEIGHT else "prior" if weight <= PRIOR_WEIGHT else "shrunk"


def _coefficient(value, sd, weight):
    return CoefficientReport(value=float(value), sd=float(sd), source=_source(weight),
                             shrinkage_weight=float(weight))


def calibration_error(fit_window, horizon, summary=CALIBRATION_SUMMARY):
    """mae_offset_s for (fit_window, horizon bucket) from a midrace summary csv path or frame.

    NaN when `summary` is None, the file is missing, or it has no such row.
    """
    from .calibration import BUCKET_LAPS
    if summary is None:
        return float("nan")
    if isinstance(summary, pd.DataFrame):
        table = summary
    elif os.path.exists(summary):
        table = pd.read_csv(summary)
    else:
        return float("nan")
    if "window" not in table.columns:
        table = table.reset_index()
    row = table[(table["window"] == fit_window) & (table["horizon_bucket"] == horizon // BUCKET_LAPS * BUCKET_LAPS)]
    return float(row["mae_offset_s"].iloc[0]) if len(row) else float("nan")


def requested_ages(snapshot: Snapshot, ages=None):
    """compound -> oldest requested tyre age: the current set on its next lap (tyre_age + 1),
    plus `ages` ({compound: age}) when given."""
    out = {snapshot.compound: snapshot.tyre_age + 1}
    for c, a in (ages or {}).items():
        c = str(c).strip().upper()
        out[c] = max(out.get(c, a), a)
    return out


def _observed(rep, compound):
    age = (rep.get("max_tyre_age") or {}).get(compound)
    return None if age is None or math.isnan(age) else float(age)


def hazard_support(hazard):
    """Compounds the fitted hazard model has observed support for (hazard.py): the keys of
    age_logistic's `a0`, the compounds with min_samples clean laps; the others borrow the
    mean a0, so their tyre risk is a guess. None when the model draws nothing ("none") or
    has no per compound table, as then it says nothing about tyre risk."""
    if hazard is None or not getattr(hazard, "draws", False):
        return None
    a0 = getattr(hazard, "a0", None)
    return set(a0) if isinstance(a0, dict) else None


def build(model, fit_window, snapshot: Snapshot, ages=None, summary=CALIBRATION_SUMMARY,
          hazard=None, min_samples=5) -> ModelReport:
    """The ModelReport of `model` (a fitted degradation model) for this snapshot.

    `hazard` is the fitted hazard model, when the simulator has one; modelled compounds
    outside its support are flagged no_hazard_support:<compound>.
    """
    rep = getattr(model, "fit_report", None)
    if rep is None:
        raise ValueError("model_report needs a degradation model with a fit_report")
    weight, sd = rep["shrinkage_weight"], rep["sd"]
    compounds = {}
    for c, (base, wear) in sorted(model.coefs.items()):
        compounds[c] = CompoundReport(
            modelled=True, n_rows=rep["n_rows"][c], n_race_rows=rep["n_race_rows_per_compound"][c],
            max_age_observed=_observed(rep, c),
            base_s=_coefficient(base, sd["base_s"][c], weight["base_s"][c]),
            wear_s_per_lap=_coefficient(wear, sd["wear_s_per_lap"][c], weight["wear_s_per_lap"][c]))
    for c, why in sorted(rep["unmodelled"].items()):
        compounds.setdefault(c, CompoundReport(modelled=False, reason=why))
    for c in snapshot.race.compounds:
        compounds.setdefault(c, CompoundReport(modelled=False, reason="no laps in the fit window"))

    fuel = None
    if weight["fuel_s_per_lap"] is not None and hasattr(model, "fuel_slope"):
        fuel = _coefficient(model.fuel_slope, sd["fuel_s_per_lap"], weight["fuel_s_per_lap"])

    flags = []
    raced = sum(1 for n in rep["n_race_rows_per_compound"].values() if n >= min_samples)
    if fuel is not None and (fuel.shrinkage_weight < WEAK_WEIGHT or raced <= 1):
        flags.append(f"wear_fuel_poorly_identified(fuel_weight={fuel.shrinkage_weight:.2f})")
    flags += [f"prior_dominated:{c}" for c, r in compounds.items()
              if r.modelled and r.wear_s_per_lap.shrinkage_weight < WEAK_WEIGHT]
    flags += [f"extrapolation:{c}" for c, age in sorted(requested_ages(snapshot, ages).items())
              if c in compounds and compounds[c].modelled and compounds[c].max_age_observed is not None
              and age > compounds[c].max_age_observed]
    hinge = rep.get("hinge")
    flags += [f"beyond_knot_unsupported:{c}" for c, age in sorted(requested_ages(snapshot, ages).items())
              if c in (hinge or {}) and hinge[c]["source"] == "dropped" and age > hinge[c]["knot"]]
    supported = hazard_support(hazard)
    if supported is not None:
        flags += [f"no_hazard_support:{c}" for c, r in compounds.items() if r.modelled and c not in supported]
    return ModelReport(
        fit_window=fit_window, compounds=compounds, fuel_s_per_lap=fuel,
        hinge=None if hinge is None else {c: dict(e) for c, e in hinge.items()},
        condition_number=rep["condition_number"], flags=flags,
        calibration_error_s=calibration_error(fit_window, snapshot.remaining_laps, summary))


def support_warnings(model, snapshot: Snapshot, plan: Plan) -> list[str]:
    """One string per stint whose end age exceeds the oldest tyre the fit saw on its compound."""
    rep = getattr(model, "fit_report", None) or {}
    out = []
    for j, s in enumerate(plan.stints):
        end = (snapshot.tyre_age if j == 0 else s.start_age) + s.laps
        seen = _observed(rep, s.compound)
        if seen is not None and end > seen:
            out.append(f"{s.compound} to age {end}, observed up to {seen:g}")
    return out
