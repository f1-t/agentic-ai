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
OUTLIER_FACTOR = 1.07  # drop laps slower than 107% of the compound's fastest (traffic/SC)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
_session = None  # lazily loaded, then reused


def _load_session():
    global _session
    if _session is None:
        fastf1.Cache.enable_cache(_CACHE_DIR)
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
def _stint_seconds(compound, n_laps, deg):
    base, slope = deg[compound]
    return sum(base + slope * k for k in range(1, n_laps + 1))


def _deg_table():
    table = {}
    for comp in ("SOFT", "MEDIUM", "HARD"):
        r = estimate_tyre_degradation(comp)
        if "error" not in r:
            table[comp] = (r["base_laptime_s"], r["deg_per_lap_s"])
    return table


def simulate_strategy(stints):
    """stints: list of [compound, n_laps]. Returns modelled total race time."""
    total = race_laps()
    deg = _deg_table()
    # normalise input
    plan = []
    for item in stints:
        try:
            comp = str(item[0]).upper()
            n = int(item[1])
        except Exception:
            return {"error": f"Each stint must be [compound, n_laps]; got {item!r}"}
        if comp not in deg:
            return {"error": f"Unknown compound '{comp}'. Modelled compounds: {sorted(deg)}"}
        if n < MIN_STINT_LAPS:
            return {"error": f"Stint of {n} laps is below the {MIN_STINT_LAPS}-lap minimum."}
        plan.append((comp, n))
    if sum(n for _, n in plan) != total:
        return {"error": f"Stint laps sum to {sum(n for _, n in plan)}, must equal race length {total}."}
    if len({c for c, _ in plan}) < 2:
        return {"error": "A valid dry strategy must use at least two different compounds."}

    num_stops = len(plan) - 1
    stint_secs = [round(_stint_seconds(c, n, deg), 1) for c, n in plan]
    total_time = round(sum(stint_secs) + PIT_LOSS_S * num_stops, 1)
    # first pit lap = laps completed in the first stint
    first_pit_lap = plan[0][1] if num_stops >= 1 else None
    return {
        "strategy": [[c, n] for c, n in plan],
        "num_stops": num_stops,
        "first_pit_lap": first_pit_lap,
        "per_stint_seconds": stint_secs,
        "pit_loss_total_s": round(PIT_LOSS_S * num_stops, 1),
        "total_time_s": total_time,
    }


# --- Ground truth (used by tests, not exposed to the agent) ------------------
def optimal_strategy(start_compound="SOFT", max_stops=2):
    """Brute-force optimum among strategies that START on `start_compound`."""
    total = race_laps()
    deg = _deg_table()
    comps = list(deg.keys())
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
