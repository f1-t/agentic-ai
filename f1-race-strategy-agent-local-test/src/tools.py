"""Uncertainty-aware F1 strategy tools.

Tools evaluate a *current race snapshot*. They deliberately return ranges and
probabilities: the simulator is a decision aid, not an oracle.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .snapshot import RaceSnapshot

# The minimum stint is rule R2 (docs/rules.md). This tool-layer value is provisional:
# SPEC 0015 has the tools read it from the snapshot's race instead. The legacy
# baseline keeps its own 10 (src/legacy_tools.py).
MIN_STINT_LAPS = 3
DEFAULT_PIT_LOSS_S = (19.0, 22.0, 25.0)
_BASE_PACE = {"SOFT": 80.4, "MEDIUM": 80.9, "HARD": 81.5}
_DEG = {"SOFT": (0.055, 0.085, 0.125), "MEDIUM": (0.035, 0.060, 0.090), "HARD": (0.025, 0.045, 0.070)}

_ACTIVE_SNAPSHOT: RaceSnapshot | None = None


def set_snapshot(snapshot: RaceSnapshot | dict[str, Any]) -> RaceSnapshot:
    global _ACTIVE_SNAPSHOT
    _ACTIVE_SNAPSHOT = snapshot if isinstance(snapshot, RaceSnapshot) else RaceSnapshot.from_dict(snapshot)
    return _ACTIVE_SNAPSHOT


def _snapshot() -> RaceSnapshot:
    if _ACTIVE_SNAPSHOT is None:
        raise ValueError("No race snapshot is active for this agent run.")
    return _ACTIVE_SNAPSHOT


def get_race_snapshot():
    """Return the immutable decision-time state supplied to this run."""
    return _snapshot().to_dict()


def estimate_tyre_degradation(compound):
    comp = str(compound).upper()
    if comp not in _DEG:
        return {"error": f"Unknown compound '{comp}'."}
    low, median, high = _DEG[comp]
    s = _snapshot()
    # A small contextual widening keeps the contract explicitly uncertain.
    width = 1.15 if s.track_status != "GREEN" else 1.0
    return {
        "compound": comp,
        "deg_per_lap_s": {"low": round(low / width, 4), "median": median, "high": round(high * width, 4)},
        "confidence": "medium",
        "basis": "historical/empirical prior adjusted to the supplied snapshot",
        "warning": "This is an estimate, not a guaranteed degradation rate.",
    }


def estimate_pit_loss():
    s = _snapshot()
    low, median, high = DEFAULT_PIT_LOSS_S
    if s.track_status in {"SC", "VSC"}:
        factor = 0.65 if s.track_status == "SC" else 0.78
        low, median, high = [x * factor for x in (low, median, high)]
    return {
        "pit_loss_s": {"low": round(low, 1), "median": round(median, 1), "high": round(high, 1)},
        "track_status": s.track_status,
        "confidence": "medium",
    }


def _validate_action(action: dict[str, Any]):
    s = _snapshot()
    kind = str(action.get("action", "")).lower()
    if kind == "stay_out":
        horizon = int(action.get("laps", 1))
        if horizon < 1 or horizon > max(1, s.laps_remaining):
            return None, f"stay_out laps must be between 1 and {max(1, s.laps_remaining)}."
        return {"action": kind, "laps": horizon}, None
    if kind == "pit":
        comp = str(action.get("compound", "")).upper()
        if comp not in s.available_compounds:
            return None, f"Compound '{comp}' is not available; choose from {s.available_compounds}."
        return {"action": kind, "compound": comp}, None
    return None, "action must be 'pit' or 'stay_out'."


def evaluate_strategy(action):
    """Evaluate one immediate decision using a distribution, never a single truth value."""
    clean, error = _validate_action(action)
    if error:
        return {"error": error, "recoverable": True, "hint": "Correct the action and call evaluate_strategy again."}
    s = _snapshot()
    current = s.compound.upper()
    deg = _DEG.get(current, _DEG["MEDIUM"])[1]
    remaining = max(1, s.laps_remaining)

    if clean["action"] == "pit":
        new = clean["compound"]
        pit = estimate_pit_loss()["pit_loss_s"]
        pace_gain = (_BASE_PACE[current] + deg * s.tyre_age_laps) - _BASE_PACE[new]
        expected_delta = pit["median"] - max(0.0, pace_gain) * min(remaining, 18)
        traffic_p = min(0.75, 0.18 + 0.06 * len(s.nearby_cars))
        improve_p = 1.0 / (1.0 + math.exp(expected_delta / 7.0))
        spread = 5.0 + 0.15 * pit["median"]
    else:
        horizon = clean["laps"]
        expected_delta = deg * (s.tyre_age_laps * horizon + horizon * (horizon + 1) / 2)
        traffic_p = min(0.65, 0.12 + 0.05 * len(s.nearby_cars))
        improve_p = max(0.05, 0.48 - 0.018 * s.tyre_age_laps)
        spread = 3.5 + 0.25 * horizon

    rain = s.rain_probability if s.rain_probability is not None else 0.0
    spread *= 1.0 + 0.7 * rain
    low = expected_delta - spread
    high = expected_delta + spread
    return {
        "action": clean,
        "relative_race_time_delta_s": {"optimistic": round(low, 1), "median": round(expected_delta, 1), "pessimistic": round(high, 1)},
        "probability_improves_outcome": round(float(np.clip(improve_p, 0.02, 0.98)), 2),
        "traffic_probability": round(traffic_p, 2),
        "uncertainty": "high" if rain >= 0.35 or s.track_status != "GREEN" else "medium",
        "assumptions": ["no unmodelled incident changes the race state", "pace/degradation remain within estimated ranges"],
        "warning": "Compare this distribution with alternatives; do not treat the median as ground truth.",
    }


TOOL_SPECS = [
    {
        "name": "get_race_snapshot",
        "description": "Read the current decision-time race snapshot: lap, position, tyre state, gaps, weather, nearby cars and recent context.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": get_race_snapshot,
    },
    {
        "name": "estimate_tyre_degradation",
        "description": "Return an uncertain low/median/high degradation estimate for a compound. It is evidence, not truth.",
        "input_schema": {"type": "object", "properties": {"compound": {"type": "string"}}, "required": ["compound"], "additionalProperties": False},
        "fn": estimate_tyre_degradation,
    },
    {
        "name": "estimate_pit_loss",
        "description": "Estimate current pit-lane loss as a low/median/high range, adjusted for track status.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": estimate_pit_loss,
    },
    {
        "name": "evaluate_strategy",
        "description": "Evaluate an immediate PIT or STAY_OUT decision probabilistically. Returns ranges, probabilities, assumptions and uncertainty; never an oracle score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "object",
                    "description": "Either {action:'pit', compound:'MEDIUM'} or {action:'stay_out', laps:3}.",
                    "properties": {"action": {"type": "string"}, "compound": {"type": "string"}, "laps": {"type": "integer"}},
                    "required": ["action"],
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "fn": evaluate_strategy,
    },
]
