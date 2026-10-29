"""Uncertainty-aware F1 strategy tools.

Tools evaluate a *current race snapshot*. They deliberately return ranges and
probabilities: the simulator is a decision aid, not an oracle.

The second half of this module is the legacy full-race path (FastF1 cache, the
deterministic simulator wrapper and the brute-force optimum). It is not exposed
to the agent; the baselines, the eval harness, grade.py and the ground truth in
data/ground_truth.json depend on it.
"""
from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Any

import fastf1
import numpy as np
from pydantic import ValidationError

from .contracts import Plan, RaceParams, Snapshot
from .sim import SimConfig, Simulator
from .sim.config import EventEffects
from .sim.data import LEGACY_OUTLIER_FACTOR, lap_frame
from .snapshot import RaceSnapshot

warnings.filterwarnings("ignore")

# Keep FastF1's cache-loading chatter out of the agent's stdout so the trajectory
# and the PASS/FAIL result are easy to read.
try:
    fastf1.set_log_level("ERROR")
except Exception:
    pass
logging.getLogger("fastf1").setLevel(logging.ERROR)

# --- Snapshot tools configuration ---------------------------------------------
MIN_STINT_LAPS = 3
DEFAULT_PIT_LOSS_S = (19.0, 22.0, 25.0)
_BASE_PACE = {"SOFT": 80.4, "MEDIUM": 80.9, "HARD": 81.5}
_DEG = {"SOFT": (0.055, 0.085, 0.125), "MEDIUM": (0.035, 0.060, 0.090), "HARD": (0.025, 0.045, 0.070)}

_ACTIVE_SNAPSHOT: RaceSnapshot | None = None

# --- Legacy full-race configuration (2021 Spanish GP) -------------------------
SEASON = 2021
GP_NAME = "Spain"
SESSION = "R"

PIT_LOSS_S = 22.0      # approx pit-lane time loss at Barcelona (seconds)
# The legacy path keeps its own minimum stint. data/ground_truth.json was generated
# with 10, and the baselines read it through get_race_info().
LEGACY_MIN_STINT_LAPS = 10
OUTLIER_FACTOR = LEGACY_OUTLIER_FACTOR  # drop laps slower than 107% of the compound's fastest (traffic/SC)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
_session = None  # lazily loaded, then reused


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


# --- Legacy full-race path (not exposed to the agent) ------------------------
def _load_session():
    global _session
    if _session is None:
        fastf1.Cache.configure(cache_dir=_CACHE_DIR)
        fastf1.Cache.offline_mode(True)  # no network: cache only
        s = fastf1.get_session(SEASON, GP_NAME, SESSION)
        s.load(telemetry=False, weather=False, messages=False)
        _session = s
    return _session


def race_laps():
    return int(_load_session().total_laps)


def get_race_info():
    s = _load_session()
    laps = s.laps
    compounds = sorted(laps["Compound"].dropna().unique().tolist())
    return {
        "race": s.event["EventName"],
        "season": SEASON,
        "total_laps": race_laps(),
        "compounds_available": compounds,
        "pit_loss_seconds": PIT_LOSS_S,
        "min_stint_laps": LEGACY_MIN_STINT_LAPS,
    }


def get_driver_stints(driver):
    s = _load_session()
    d = s.laps.pick_drivers(driver)
    if len(d) == 0:
        return {"error": f"No laps found for driver '{driver}'. "
                         f"Use a 3-letter code like HAM or VER."}
    stints = []
    for stint_no, grp in d.groupby("Stint"):
        laptimes = grp["LapTime"].dt.total_seconds().dropna()
        stints.append({
            "stint": int(stint_no),
            "compound": grp["Compound"].iloc[0],
            "laps": int(len(grp)),
            "start_lap": int(grp["LapNumber"].min()),
            "end_lap": int(grp["LapNumber"].max()),
            "avg_laptime_s": round(float(laptimes.mean()), 3) if len(laptimes) else None,
        })
    stints.sort(key=lambda x: x["start_lap"])
    return {
        "driver": driver,
        "starting_compound": stints[0]["compound"] if stints else None,
        "num_stops": len(stints) - 1,
        "stints": stints,
    }


def fit_tyre_degradation(compound):
    """Legacy per-compound linear fit from this race's green-flag laps. Formerly
    estimate_tyre_degradation; that name is now the snapshot tool above."""
    compound = compound.upper()
    s = _load_session()
    laps = s.laps
    d = laps[(laps["Compound"] == compound) & (laps["TrackStatus"] == "1")].copy()
    d = d[d["PitInTime"].isna() & d["PitOutTime"].isna()]  # drop in/out laps
    d["t"] = d["LapTime"].dt.total_seconds()
    d = d.dropna(subset=["t", "TyreLife"])
    if len(d) >= 1:
        d = d[d["t"] < d["t"].min() * OUTLIER_FACTOR]  # drop traffic/SC laps
    if len(d) < 5:
        return {"error": f"Not enough clean laps to model compound '{compound}'. "
                         f"Available compounds: {sorted(laps['Compound'].dropna().unique().tolist())}"}
    x = d["TyreLife"].astype(float).values
    y = d["t"].values
    slope, base = np.polyfit(x, y, 1)
    return {
        "compound": compound,
        "base_laptime_s": round(float(base), 3),   # modelled lap time at zero tyre age
        "deg_per_lap_s": round(float(slope), 4),    # seconds lost per lap of tyre age
        "n_samples": int(len(d)),
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
            min_stint_laps=LEGACY_MIN_STINT_LAPS,
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
        for n1 in range(LEGACY_MIN_STINT_LAPS, total - LEGACY_MIN_STINT_LAPS + 1):
            consider([(start, n1), (c2, total - n1)])
    # 2-stop
    if max_stops >= 2:
        for c2 in comps:
            for c3 in comps:
                for n1 in range(LEGACY_MIN_STINT_LAPS, total - 2 * LEGACY_MIN_STINT_LAPS + 1):
                    for n2 in range(LEGACY_MIN_STINT_LAPS, total - n1 - LEGACY_MIN_STINT_LAPS + 1):
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
