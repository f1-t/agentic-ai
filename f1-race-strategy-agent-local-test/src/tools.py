"""F1 strategy tools + a deterministic strategy simulator.

These are the tools the agent (src/agent.py) can call. Everything reads from the
committed FastF1 cache in offline mode, so there is no network dependency on race
data. The strategy simulator is a simplified, deterministic tyre-degradation model
-- it is the agent's "what-if" evaluator and also the source of ground truth.
"""
import logging
import os
import warnings

import fastf1
import numpy as np
from pydantic import ValidationError

from .contracts import Plan, RaceParams, Snapshot
from .sim import SimConfig, Simulator
from .sim.data import LEGACY_OUTLIER_FACTOR, lap_frame

warnings.filterwarnings("ignore")

# Keep FastF1's cache-loading chatter out of the agent's stdout so the trajectory
# and the PASS/FAIL result are easy to read.
try:
    fastf1.set_log_level("ERROR")
except Exception:
    pass
logging.getLogger("fastf1").setLevel(logging.ERROR)

# --- Race configuration (2021 Spanish GP) ------------------------------------
SEASON = 2021
GP_NAME = "Spain"
SESSION = "R"

PIT_LOSS_S = 22.0      # approx pit-lane time loss at Barcelona (seconds)
MIN_STINT_LAPS = 10    # a realistic minimum stint length
OUTLIER_FACTOR = LEGACY_OUTLIER_FACTOR  # drop laps slower than 107% of the compound's fastest (traffic/SC)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
_session = None  # lazily loaded, then reused


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


# --- Tool 1: race info -------------------------------------------------------
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
        "min_stint_laps": MIN_STINT_LAPS,
    }


# --- Tool 2: a driver's actual stints ----------------------------------------
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


# --- Tool 3: tyre degradation model ------------------------------------------
def estimate_tyre_degradation(compound):
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
            min_stint_laps=MIN_STINT_LAPS,
            compounds=sorted(laps["Compound"].dropna().unique().tolist()),
        )
    return _params


def _get_simulator():
    global _simulator
    if _simulator is None:
        # baseline filter and model, so ground truth and the graded test keep their numbers
        frame = lap_frame(_load_session().laps, OUTLIER_FACTOR)
        _simulator = Simulator(frame, SimConfig(degradation={"name": "linear"}))
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
        "name": "get_race_info",
        "description": "Get race name, total laps, tyre compounds available, pit-loss and the minimum stint length. Call this first.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": lambda: get_race_info(),
    },
    {
        "name": "get_driver_stints",
        "description": "Get a driver's ACTUAL stints in this race (compound, laps, average lap time) and their starting compound. Driver is a 3-letter code, e.g. HAM.",
        "input_schema": {
            "type": "object",
            "properties": {"driver": {"type": "string", "description": "3-letter driver code, e.g. HAM"}},
            "required": ["driver"],
            "additionalProperties": False,
        },
        "fn": lambda driver: get_driver_stints(driver),
    },
    {
        "name": "estimate_tyre_degradation",
        "description": "Fit a linear tyre-degradation model for a compound from this race's green-flag laps. Returns base lap time (s) and degradation (s per lap of tyre age).",
        "input_schema": {
            "type": "object",
            "properties": {"compound": {"type": "string", "description": "SOFT, MEDIUM or HARD"}},
            "required": ["compound"],
            "additionalProperties": False,
        },
        "fn": lambda compound: estimate_tyre_degradation(compound),
    },
    {
        "name": "simulate_strategy",
        "description": "Estimate total race time for a proposed strategy. Input is a list of [compound, n_laps] stints that must sum to the race length and use >=2 compounds. Use this to compare candidate strategies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "stints": {
                    "type": "array",
                    "description": "e.g. [[\"SOFT\", 28], [\"MEDIUM\", 38]]",
                    "items": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}, {"type": "integer"}],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                }
            },
            "required": ["stints"],
            "additionalProperties": False,
        },
        "fn": lambda stints: simulate_strategy(stints),
    },
]
