"""Provenance of everything the policy side reads that was estimated from other races, or set a
priori (docs/eval.md "Information contract").

POLICY_SIDE maps a name to an Artifact: where it lives, its kind and, per race, the races it reads
when pricing that race:

  as_of      a table of per-race observations read as of the race: only races that started
             strictly before it (race_ids.as_of), an a priori fallback with no earlier race
  a_priori   an engineering prior stated in code, fitted on no race (reads no race)
  not_as_of  fitted on a fixed race set; off by default and refused for a race not after all
             of them unless forced (the learned calibrator's artifacts)

races_for(name, race_id) is what pricing `race_id` reads; recorded(name) every race the
artifact records at all. tests/future_blindness.py checks both for every name. The world (the
judge, the hindsight reference) may read whole-set tables and is not listed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Artifact:
    where: str
    kind: str                                  # "as_of", "a_priori" or "not_as_of"
    recorded: Callable[[], list]               # every race it records
    races_for: Callable[[str], list]           # the races pricing race_id reads


def _none():
    return []


def _none_for(race_id):
    return []


def _proxy_recorded():
    from . import proxy
    from .config import CompoundProxy
    return proxy.history_races(CompoundProxy())


def _proxy_races(race_id):
    from .race_history import earlier_races
    return earlier_races(race_id, _proxy_recorded())


def _history_recorded():
    from .race_history import load_history
    return sorted(map(str, load_history()["race_id"]))


def _history_for(race_id):
    from .race_history import earlier_races
    return earlier_races(race_id, _history_recorded())


def _periods_recorded():
    from .race_history import load_periods
    return sorted(set(map(str, load_periods()["race_id"])))


def _periods_for(race_id):
    from .race_history import as_of_event_rates
    return as_of_event_rates(race_id)["races"]


def _summary_recorded():
    from .model_report import CALIBRATION_SUMMARY_RACES
    return list(CALIBRATION_SUMMARY_RACES)


def _summary_for(race_id):
    from .race_history import earlier_races
    races = _summary_recorded()
    return races if len(earlier_races(race_id, races)) == len(races) else []


def _pit_loss_recorded():
    from .config import PitLossModel
    return sorted((PitLossModel().prior or {}).get("races", {}))


def _pit_loss_for(race_id):
    from . import pit_loss
    from .config import PitLossModel
    return list(pit_loss.pit_loss_prior(race_id, PitLossModel())["races"])


def _tyre_recorded():
    from . import tyre_history
    from .cliff import load_history
    return sorted(set(load_history().get("races", {})) | set(map(str, tyre_history.load_history()["race_id"])))


def _tyre_for(race_id):
    from .race_history import earlier_races
    return earlier_races(race_id, _tyre_recorded())


def _learned_recorded():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    out = set()
    for m in root.glob("**/learned*/**/manifest.json"):
        try:
            out |= set(json.loads(m.read_text()).get("races", []))
        except (OSError, ValueError):
            continue
    return sorted(out)


POLICY_SIDE = {
    "compound_steps": Artifact(
        "configs/sim/compound_steps.json per_race and validation_rows: the proxy's tier-5 step, blend weights and sd "
        "floor, recomputed as of the race (proxy.as_of_config); no earlier race: no step, equal weights, "
        "proxy.PRIOR_SD_FLOOR", "as_of", _proxy_recorded, _proxy_races),
    "calibration_summary": Artifact(
        "reports/calibration/midrace_summary.csv (2021 Spain): calibration_error_s in the model report and "
        "evaluate_strategy, NaN for a race not after every summary race (model_report.calibration_error)",
        "as_of", _summary_recorded, _summary_for),
    "event_rates": Artifact(
        "configs/sim/neutralised_history.csv: the belief's SC/VSC rates (events.RateEvents as_of, "
        "race_history.as_of_event_rates) shrunk toward the a priori race_history.EVENT_PRIOR",
        "as_of", _periods_recorded, _periods_for),
    "pre_race_base_delta": Artifact(
        "configs/sim/race_history.csv base_delta_s: the practice-to-race base level (as_of_history calibrator), "
        "a priori race_history.A_PRIORI with no earlier race", "as_of", _history_recorded, _history_for),
    "fuel_prior": Artifact(
        "configs/sim/race_history.csv target: the circuit-history fuel prior (fuel_prior.circuit_history_prediction "
        "as of the race), the a priori physics prior (fuel_prior.practice_longrun_prediction) with fewer than two "
        "earlier races", "as_of", _history_recorded, _history_for),
    "event_effects": Artifact(
        "configs/sim/race_history.csv sc_lap_factor, lap1_extra_s: the belief's SC lap factor and lap-1 extra "
        "(as_of_history calibrator), a priori race_history.A_PRIORI with no earlier race",
        "as_of", _history_recorded, _history_for),
    "shrinkage": Artifact(
        "SimConfig.shrinkage base_s 0.5 and wear_s_per_lap 0.03 (and the hinge tau): engineering priors set in "
        "a8d9d3a; under as_of_history the base and fuel taus are the as-of sds instead",
        "a_priori", _none, _none_for),
    "pit_loss_prior": Artifact(
        "configs/sim/pit_loss_prior.json and the cached races: the as-of pit loss prior (pit_loss.pit_loss_prior)",
        "as_of", _pit_loss_recorded, _pit_loss_for),
    "tyre_history": Artifact(
        "configs/sim/tyre_life_history.json and data/derived/tyre_history*.csv: the tyre-life prior "
        "(cliff.life_prior, tyre_history.tyre_life_prior), races strictly before the race",
        "as_of", _tyre_recorded, _tyre_for),
    "learned_calibrator": Artifact(
        "reports/calibration/learned artifacts (calibrator 'trained', configs/sim/belief_learned.yaml): fitted on a "
        "fixed race set, off by default; refused for a race not after all its training races unless "
        "force_not_as_of", "not_as_of", _learned_recorded, _none_for),
}


def recorded(name: str) -> list:
    return list(POLICY_SIDE[name].recorded())


def races_for(name: str, race_id: str) -> list:
    return list(POLICY_SIDE[name].races_for(race_id))
