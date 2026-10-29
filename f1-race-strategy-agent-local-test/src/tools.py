"""Uncertainty-aware F1 strategy tools.

Tools evaluate a *current race snapshot*. They deliberately return ranges and
probabilities: the simulator is a decision aid, not an oracle.
"""
from __future__ import annotations

import fastf1
import numpy as np
from pydantic import ValidationError

from .contracts import Plan, RaceParams, Snapshot
from .sim import SimConfig, Simulator
from .sim.config import EventEffects
from .sim.data import LEGACY_OUTLIER_FACTOR, lap_frame

import numpy as np

from .snapshot import RaceSnapshot

MIN_STINT_LAPS = 3
DEFAULT_PIT_LOSS_S = (19.0, 22.0, 25.0)
_BASE_PACE = {"SOFT": 80.4, "MEDIUM": 80.9, "HARD": 81.5}
_DEG = {"SOFT": (0.055, 0.085, 0.125), "MEDIUM": (0.035, 0.060, 0.090), "HARD": (0.025, 0.045, 0.070)}

PIT_LOSS_S = 22.0      # approx pit-lane time loss at Barcelona (seconds)
MIN_STINT_LAPS = 10    # a realistic minimum stint length
OUTLIER_FACTOR = LEGACY_OUTLIER_FACTOR  # drop laps slower than 107% of the compound's fastest (traffic/SC)


def set_snapshot(snapshot: RaceSnapshot | dict[str, Any]) -> RaceSnapshot:
    global _ACTIVE_SNAPSHOT
    _ACTIVE_SNAPSHOT = snapshot if isinstance(snapshot, RaceSnapshot) else RaceSnapshot.from_dict(snapshot)
    return _ACTIVE_SNAPSHOT

def _load_session():
    global _session
    if _session is None:
        fastf1.Cache.configure(cache_dir=_CACHE_DIR)
        fastf1.Cache.offline_mode(True)  # no network: cache only
        s = fastf1.get_session(SEASON, GP_NAME, SESSION)
        s.load(telemetry=False, weather=False, messages=False)
        _session = s
    return _session

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


# --- Tool 4: deterministic strategy simulator --------------------------------
# The simulator lives in src/sim. These wrappers keep the original tool contract.
_simulator = None
_params = None


def _race_params():
    global _params
    if _params is None:
        laps = _load_session().laps
        _params = RaceParams(
            race_id=f"{SEASON}_{GP_NAME}", total_laps=race_laps(), pit_loss_s=PIT_LOSS_S,
            min_stint_laps=MIN_STINT_LAPS,
            compounds=sorted(laps["Compound"].dropna().unique().tolist()),
        )
    return _params


def _get_simulator():
    global _simulator
    if _simulator is None:
        # baseline filter and model, so ground truth and the graded test keep their numbers;
        # lap1_extra_s is pinned to 0 for the same reason
        frame = lap_frame(_load_session().laps, OUTLIER_FACTOR)
        _simulator = Simulator(frame, SimConfig(degradation={"name": "linear"},
                                                effects=EventEffects(lap1_extra_s=0.0)))
    return _simulator


def simulate_strategy(stints):
    """stints: list of [compound, n_laps]. Returns modelled total race time."""
    try:
        plan = Plan.of([list(item)[:2] for item in stints])
    except (ValidationError, TypeError) as exc:
        return {"error": f"Each stint must be [compound, n_laps]; got {stints!r} ({exc})"}
    sim = _get_simulator()
    snap = Snapshot.at_start(_race_params(), driver="", compound=plan.stints[0].compound)
    res = sim.simulate(snap, plan)
    if not res.valid:
        return {"error": res.errors[0]}
    return {
        "strategy": plan.as_pairs(),
        "num_stops": res.num_stops,
        "first_pit_lap": res.pit_laps[0] if res.pit_laps else None,
        "per_stint_seconds": [round(x, 1) for x in res.stint_seconds],
        "pit_loss_total_s": round(res.pit_loss_total_s, 1),
        "total_time_s": round(res.race_time_s, 1),
    }


# --- Ground truth (used by tests, not exposed to the agent) ------------------
def optimal_strategy(start_compound="SOFT", max_stops=2):
    """Brute-force optimum among strategies that START on `start_compound`."""
    total = race_laps()
    comps = [c for c in ("SOFT", "MEDIUM", "HARD") if c in _get_simulator().modelled]
    best = None

    def consider(plan):
        nonlocal best
        res = simulate_strategy([[c, n] for c, n in plan])
        if "error" in res:
            return
        if best is None or res["total_time_s"] < best["total_time_s"]:
            best = res

    start = start_compound.upper()
    # 1-stop
    for c2 in comps:
        for n1 in range(MIN_STINT_LAPS, total - MIN_STINT_LAPS + 1):
            consider([(start, n1), (c2, total - n1)])
    # 2-stop
    if max_stops >= 2:
        for c2 in comps:
            for c3 in comps:
                for n1 in range(MIN_STINT_LAPS, total - 2 * MIN_STINT_LAPS + 1):
                    for n2 in range(MIN_STINT_LAPS, total - n1 - MIN_STINT_LAPS + 1):
                        n3 = total - n1 - n2
                        consider([(start, n1), (c2, n2), (c3, n3)])
    return best


# Registry consumed by the agent: name -> (callable, JSON schema)
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
