"""Simulator fidelity scorecard: judge against reality, predictive against judge, components, ablation.

    python -m src.sim.fidelity run    --races 2021_04_spain 2023_07_spain --out reports/calibration/fidelity
    python -m src.sim.fidelity report --out reports/calibration/fidelity
    python -m src.sim.fidelity ablate --races ... --out reports/calibration/fidelity [--spec configs/sim/fidelity/ablation.yaml]
    python -m src.sim.fidelity ledger [--world default] [--belief configs/sim/belief.yaml]

There are two simulators (docs/eval.md "Who sees what" and "Simulator fidelity"). The JUDGE (the
eval's "world") scores every plan and builds the reference; it should be as close as possible to
what really happened while using as little hindsight as it can. The PREDICTIVE simulator (the
"belief") is what the agent, its tools and the baselines plan with at a snapshot; it must use no
hindsight, and should be as close as possible to the judge under that restriction. This module
measures both distances for any configuration (world config + belief config, files and overlays)
and never changes a lap time. Finishing ORDER is the headline everywhere: order agreement first
(close-competitor pairs, then Spearman and position error), then relative time (gaps between
cars, common offset removed), then absolute time.

Sections (docs/eval.md "Simulator fidelity" defines every measure in one line):

  O  ORDER AND TIME, three versions side by side: (1) the judge as configured, in-sample and
     leave-one-driver-out; (2) the judge with each whole-race input of its hindsight ledger
     replaced by its no-hindsight substitute, singly and all together; (3) the predictive
     simulator replaying every car's real remaining strategy from laps 0, 10, 20, 30, 40.
     Every row says what the car's pace and the tyre parameters were fitted on.
  A  JUDGE versus REALITY: whole-race replays (A1), finishing order (A2), stints, lap residuals by
     tyre age, stop gain, pit loss, neutralised laps (A3), the lap-0 decision (A4), each in-sample
     and out of sample (A5: leave-one-driver-out, leave-one-stint-out), the hindsight ledger (A6)
     and the paired no-hindsight substitutes.
  B  PREDICTIVE versus JUDGE on a fixed plan panel at a snapshot grid: ranking (B2) first, then
     remaining time (B1), uncertainty calibration (B3), predictive versus reality (B4, the G
     measure) and the future-blindness harness when the branch has it (B5).
  D  COMPONENT TARGETS: per model component, its prediction from allowed information against the
     whole-race (hindsight) measurement of the same quantity. A registry (COMPONENTS): a branch
     adding a factor adds one Component.
  C  ABLATION: the scorecard for a base, each factor switch alone, and an "all on" configuration;
     one table of headline changes with race-level bootstrap intervals and the ledger change.

Results are cached per (race, section, config hash, code version, options) under <out>/.cache,
so an ablation never recomputes the base. Races in data/eval_set.json are refused (no override).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..contracts import Plan
from . import calibrate, calibration, cross_race, rules
from .config import SimConfig
from .engine import Simulator
from .events import Scenario as EventScenario
from .race_ids import canonical_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "reports" / "calibration" / "fidelity"
BELIEF_FILE = ROOT / "configs" / "sim" / "belief.yaml"
ABLATION_SPEC = ROOT / "configs" / "sim" / "fidelity" / "ablation.yaml"
# the calibration races: the six of the first run, and the eligible, non-held-out 2021-22 weekends
# cached since (never `auto`: a race is added here on purpose)
CALIBRATION_RACES = ["2021_04_spain", "2023_07_spain", "2023_01_bahrain", "2019_10_united_kingdom",
                     "2022_02_saudi_arabia", "2024_07_emilia_romagna",
                     "2021_01_bahrain", "2021_09_austrian", "2021_20_qatar", "2021_22_united_arab_emirates",
                     "2022_01_bahrain", "2022_06_spain", "2022_11_austria", "2022_08_azerbaijan", "2022_09_canada",
                     "2022_12_france", "2022_20_mexico", "2022_21_brazil", "2022_22_united_arab_emirates"]
# never loaded, run or inspected here, whatever data/eval_set.json says (the d4 races); every race
# data/eval_set.json lists is refused too (guard)
NEVER = frozenset({"2019_06_monaco", "2020_16_sakhir", "2025_11_austria", "2024_11_austria"})
DEFAULT_WORLD = "default"   # the race's own config from jobs.load_race (the eval's default world)
X_LEDGER = "x_ledger"       # config-file key declaring ledger entries; removed before SimConfig sees it

ORDER_KS = (0, 10, 20, 30, 40)   # version 3 start laps, and the B4 (G) laps
CLOSE_PAIR_S = 10.0              # close competitors: same laps, real finishing gap under this
REG_RULES = ["first_stint_matches", "lap_sum", "known_compounds", "min_stint", "two_compounds"]
AGE_BUCKETS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 1000)
LATE_STINT_LAPS = 5
MIN_OFFSET_LAPS = calibration.MIN_OFFSET_LAPS
COVERAGE_SAMPLES = 100
N_BOOT = 2000
Z90 = 1.6448536269514722


def _sha12(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:12]


# The cache key's code version: the simulator's physics version and this module's source up to
# the "# summaries" banner (the sections that compute the cached tables). Summaries and writing
# below it read the cached tables, so changing them does not recompute anything.
SUMMARIES_BANNER = b"\n# summaries\n"
CODE_VERSION = f"{calibration.SIM_CODE_VERSION}-{_sha12(Path(__file__).read_bytes().split(SUMMARIES_BANNER)[0])}"


# --- small numeric helpers (no RuntimeWarning on empty input) -----------------------------------
def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _mae(x) -> float:
    return _mean(np.abs(np.asarray(x, dtype=float)))


def _spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    ra, rb = a[ok].rank(), b[ok].rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _share(mask) -> float:
    m = np.asarray(mask, dtype=float)
    m = m[np.isfinite(m)]
    return float(m.mean()) if len(m) else float("nan")


# --- configurations ------------------------------------------------------------------------------
def read_config_file(path) -> tuple[dict, list]:
    """(config dict, declared ledger entries) of a yaml or json SimConfig file. The X_LEDGER key
    (a list of {id, input, tag, note}) declares hindsight-ledger entries the code cannot derive;
    it is removed before SimConfig validates the rest."""
    import yaml
    text = Path(path).read_text()
    raw = (yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)) or {}
    declared = list(raw.pop(X_LEDGER, None) or [])
    return raw, declared


def merge(base: dict, overlay: dict) -> dict:
    """`overlay` deep-merged onto `base`, except that an overlay naming another events model replaces
    the events block: event params belong to their model (hybrid's lap_times is no RateEvents
    field), so switching the default hybrid world to "rate" or "none" must not carry them over."""
    out = cross_race._merge(copy.deepcopy(base), copy.deepcopy(overlay or {}))
    ev, was = (overlay or {}).get("events"), (base or {}).get("events")
    if isinstance(ev, dict) and isinstance(was, dict) and ev.get("name", was.get("name")) != was.get("name"):
        out["events"] = {**{k: v for k, v in out["events"].items() if k not in ("name", "params")},
                         "name": ev["name"], "params": copy.deepcopy(ev.get("params") or {})}
    return out


@dataclass
class Spec:
    """One simulator configuration under test. `world` None is the eval's default world
    (score.default_world_config of the race's own config). `world_race` overrides the judge's
    race constants: "pit_loss_s" is the pit loss the judge charges (a number, or "prior" for the
    pre-race prior from earlier races; absent: the world's whole-race value, score.world_pit_loss),
    any other key a RaceParams field on the judge's snapshots."""
    label: str = "main"
    world: dict | None = None
    belief: dict = field(default_factory=dict)
    world_overlay: dict = field(default_factory=dict)
    belief_overlay: dict = field(default_factory=dict)
    world_race: dict = field(default_factory=dict)
    declared: dict = field(default_factory=lambda: {"judge": [], "predictive": []})

    def world_config(self, data=None) -> SimConfig:
        from ..eval.score import default_world_config
        from . import config as _config
        world_default = getattr(_config, "world_default", lambda c: c)   # the eval world's wear fit
        base = self.world if self.world is not None else default_world_config(
            world_default(data.config if data is not None else SimConfig())).model_dump()
        return SimConfig.model_validate(merge(base, self.world_overlay))

    def belief_config(self) -> SimConfig:
        return SimConfig.model_validate(merge(self.belief, self.belief_overlay))

    def with_factor(self, label, world=None, belief=None, world_race=None) -> "Spec":
        return replace(self, label=label, world_overlay=merge(self.world_overlay, world or {}),
                       belief_overlay=merge(self.belief_overlay, belief or {}),
                       world_race={**self.world_race, **(world_race or {})},
                       declared=copy.deepcopy(self.declared))


def make_spec(world=DEFAULT_WORLD, belief=BELIEF_FILE, world_overlays=(), belief_overlays=(), label="main") -> Spec:
    declared = {"judge": [], "predictive": []}
    w = None
    if world not in (None, DEFAULT_WORLD):
        w, declared["judge"] = read_config_file(world)
    b, declared["predictive"] = read_config_file(belief)
    spec = Spec(label=label, world=w, belief=b, declared=declared)
    for p in world_overlays:
        o, d = read_config_file(p)
        spec.world_overlay = merge(spec.world_overlay, o)
        spec.declared["judge"] += d
    for p in belief_overlays:
        o, d = read_config_file(p)
        spec.belief_overlay = merge(spec.belief_overlay, o)
        spec.declared["predictive"] += d
    return spec


def config_key(cfg: SimConfig) -> str:
    """sha12 over the whole SimConfig, fit_window included (calibration.config_hash leaves it out)."""
    return _sha12(json.dumps(cfg.model_dump(mode="json"), sort_keys=True))


# --- races and the held-out guard -------------------------------------------------------------
def guard(races) -> list[str]:
    """Canonical ids; refuses (ValueError) any race data/eval_set.json holds out and the NEVER
    races, before anything about them is loaded. There is no override."""
    ids = [canonical_race_id(r) for r in races]
    cross_race.guard_held_out(ids, allow=False)
    bad = sorted(set(ids) & NEVER)
    if bad:
        raise ValueError(f"{bad} are evaluation races: the fidelity scorecard never loads them")
    return ids


def load(race_id, profile=None):
    from .compute import load_profile
    return cross_race.load_race(race_id, profile or load_profile())


def _finishers(frame) -> list[str]:
    """The cars compared: at least 90% of the winner's laps (calibration._finisher_ids). A
    stand-in for "classified"; retirements have no comparable race time."""
    return calibration._finisher_ids(frame, 0.9)


def _sample_drivers(data, n, full_distance=False) -> list[str]:
    """`n` of the eligible drivers (eval.scenarios.eligible_drivers), seeded by race id. With
    `full_distance`, only cars that ran the scheduled distance, so a snapshot's remaining laps
    (scheduled, snapshot_as_of) match their real remaining strategy."""
    from ..eval.scenarios import eligible_drivers
    pool = eligible_drivers(data.frame, data.race.total_laps)
    if full_distance:
        laps = data.frame.groupby("driver")["lap"].max()
        pool = [d for d in pool if int(laps.get(d, 0)) == int(data.race.total_laps)]
    if n is None or n >= len(pool):
        return pool
    rng = np.random.default_rng(int(_sha12(data.race_id), 16))
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def _judge_race(data, spec: Spec):
    update = {k: v for k, v in spec.world_race.items() if k != "pit_loss_s"}
    return data.race.model_copy(update=update) if update else data.race


def _sim(data, cfg: SimConfig, calibrator=None, frame=None) -> Simulator:
    return Simulator(data.frame if frame is None else frame, cfg, data.practice,
                     qualifying=getattr(data, "qualifying", None), calibrator=calibrator)


def judge_pit_loss(data, spec: Spec, world: Simulator) -> float:
    """The pit loss the judge charges: spec.world_race["pit_loss_s"] when set ("prior": the
    pre-race prior from earlier races, calibration.pre_race_pit_loss), else the world's
    whole-race value as the eval gives it (score.world_pit_loss)."""
    from ..eval.score import world_pit_loss
    v = spec.world_race.get("pit_loss_s")
    if v == "prior":
        return float(calibration.pre_race_pit_loss(data.race_id))
    if v is not None:
        return float(v)
    return float(world_pit_loss(data, world))


def _judge(data, spec: Spec, cfg: SimConfig | None = None, frame=None, pit_loss_s=None) -> Simulator:
    """The judge as the eval builds it: the world config with the race's whole-race pit loss
    (Simulator.race_pit_loss_s), unless the spec substitutes it."""
    world = _sim(data, cfg or spec.world_config(data), frame=frame)
    world.race_pit_loss_s = judge_pit_loss(data, spec, world) if pit_loss_s is None else float(pit_loss_s)
    return world


def replay_events(cfg: SimConfig) -> str:
    """How a whole-race replay gets its neutralisations: "recorded" (the judge's own recorded
    events, with their recorded lap times) when the judge models them (events recorded or
    hybrid: from lap 0 a hybrid world is the recorded one for the periods that happened), else
    "scenario" (the car's recorded SC/VSC laps as a fixed scenario, priced by the factors)."""
    return "recorded" if cfg.events.name in ("recorded", "hybrid") else "scenario"


def _replay_config(cfg: SimConfig) -> SimConfig:
    """A hybrid judge replays as "recorded" (whole race from lap 0, so every period is a recorded
    one); others as they are."""
    if cfg.events.name == "hybrid":
        return cfg.model_copy(update={"events": cfg.events.model_copy(update={"name": "recorded"})})
    return cfg


# --- options and the cache ------------------------------------------------------------------------
@dataclass
class Options:
    b_drivers: int = 6          # drivers sampled per race for the B grid
    b_every: int = 5            # B snapshot every this many laps
    # draws per simulation in B and version 3. One draw of the belief (a posterior parameter draw,
    # lap noise and drawn SC/VSC periods) is a random draw, not its prediction, and with one seed
    # it is the same kind of draw at every snapshot, so a single draw biases B1; 20 keep the mean
    # stable at small cost (the samples are vectorised). 0 = COVERAGE_SAMPLES.
    samples: int = 20
    n_one: int = 4              # one-stop laps in the panel (per compound)
    n_two: int = 3              # two-stop first-stop laps in the panel (per compound pair)
    lodo: bool = True
    loso: bool = True
    grid_references: bool = True    # A4 also over the judge's reference at every B snapshot
    hindsight_variants: bool = True # A paired: each ledger input replaced by its substitute
    components: bool = True
    # also run src/sim/field_replay.py's whole-field replay (every car together, traffic between
    # them) and report its own summary; not cached, so off unless asked (--field-replay)
    field_replay: bool = False
    seed: int = 0
    max_stops: int = 2

    def key(self, section) -> str:
        keep = {"A": ("lodo", "loso", "max_stops", "seed"),
                "H": ("max_stops", "seed"),
                "B": ("b_drivers", "b_every", "samples", "n_one", "n_two", "grid_references", "seed", "max_stops"),
                "D": ("seed",)}[section]
        return _sha12(json.dumps({k: getattr(self, k) for k in keep}, sort_keys=True))


def _cache_path(cache_dir, race_id, section, *parts) -> Path:
    return Path(cache_dir) / race_id / f"{section}_{_sha12('|'.join(map(str, parts)))}.pkl"


def _cached(cache_dir, race_id, section, parts, fn):
    if cache_dir is None:
        return fn()
    path = _cache_path(cache_dir, race_id, section, *parts, CODE_VERSION)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


# =================================================================================================
# A6: the hindsight ledger
# =================================================================================================
TAGS = ("this race, whole", "this race, to snapshot", "weekend (pre-race)", "history", "regulation", "a priori",
        "unclassified")
_W, _K, _WE, _H, _R, _P, _U = TAGS


def _window_tag(cfg) -> str:
    return {"full": _W, "to_snapshot": _K, "pre_race": _WE}[cfg.fit_window]


def ledger(cfg: SimConfig, role: str, declared=()) -> list[dict]:
    """Every input the configuration reads, tagged (TAGS). Derived from the config and the code
    paths of this branch; `declared` entries (X_LEDGER, {id, input, tag, note}) override by id or
    add. A config field that differs from SimConfig()'s default and has no rule here is listed as
    "unclassified" so it is visible. `substitute` names a SUBSTITUTES entry: the judge variant that
    replaces this input with a no-hindsight one (Section A paired)."""
    out = []

    def add(id_, inp, tag, note="", substitute=None):
        out.append({"role": role, "id": id_, "input": inp, "tag": tag, "note": note, "substitute": substitute,
                    "source": "derived"})
    wt = _window_tag(cfg)
    windowed = cfg.fit_window != "full"
    pv = getattr(cfg, "policy_view", None)
    cut = getattr(pv, "window_cut", "lap")
    add("fit_window", f"degradation fit rows (fit_window {cfg.fit_window})", wt,
        {"full": "every race lap of every car",
         "to_snapshot": f"race rows visible at the snapshot (cut by {'session time' if cut == 'time' else 'lap number'}), "
                        "shrunk toward the practice prior",
         "pre_race": "practice long runs only"}[cfg.fit_window],
        "pre_race_fit" if cfg.fit_window == "full" else None)
    # the lap filter: the full window reads the lap frame's whole-race robust filter; a windowed fit
    # recomputes it on the visible rows under policy_view.clean "as_of" (data.as_of_columns)
    as_of_clean = windowed and getattr(pv, "clean", "race") == "as_of"
    add("clean_flag", "lap filter (clean flag, pace residual, driver offset)",
        _K if as_of_clean else (_WE if cfg.fit_window == "pre_race" else _W),
        "recomputed on the rows visible at the snapshot" if as_of_clean else
        "data.lap_frame's robust pace fit over every lap of the race")
    add("ref_lap_s", "SC lap time base ref_lap_s (median clean lap of the fit rows)",
        _K if as_of_clean else (_WE if cfg.fit_window == "pre_race" else _W))
    if role == "judge":
        add("pit_loss", "pit loss (Simulator.race_pit_loss_s, eval.score.world_pit_loss)", _W,
            f"the race's whole-race value: pit_loss.whole_race, the median green stop against the full fit "
            f"(pit_loss.world {cfg.pit_loss.world!r})", "pit_loss_prior")
    else:
        src = getattr(cfg.pit_loss, "source", "snapshot")
        add("pit_loss", f"pit loss (pit_loss.source {src})", _K if src == "asof" else _H,
            "the prior from earlier races (configs/sim/pit_loss_prior.json and the cache) updated with this race's "
            "green stops seen by the snapshot" if src == "asof" else
            "the snapshot's pre-race prior from earlier races (calibration.pre_race_pit_loss)")
        add("pit_loss_prior", "pit loss prior table (configs/sim/pit_loss_prior.json, earlier races only)", _H)
    add("race_length", "race length (scheduled distance: session.total_laps in the scenario snapshots, "
        "calibration.snapshot_as_of)", _R,
        "the scorecard's own whole-race replays run each car's real laps: a measurement, not an input")
    comp_src = getattr(calibrate, "DEFAULT_COMPOUND_SOURCE", "raced")
    add("compound_list", f"compound list (RaceParams.compounds: {comp_src})", _R if comp_src == "nominated" else _W,
        "the nominated dry three" if comp_src == "nominated" else "every compound in the race frame")
    if "stint_limit" in cfg.rules and cfg.stint_limit_margin_laps is not None:
        add("stint_limit", f"R8 stint limit (rank-{cfg.stint_limit_rank} oldest age per compound + "
            f"{cfg.stint_limit_margin_laps})", _W, "Simulator.stint_limits from the full race frame", "r8_off")
    wc = getattr(cfg, "wear_cliff", None)
    if wc is not None and wc.enabled:
        if role == "judge" and cfg.fit_window == "full":
            edge = wc.world_edge
            add("cliff_edge", f"wear-cliff support edge (world_edge {edge})",
                {"history_only": _H, "history": _W, "race": _W}[edge],
                {"history_only": "the historical tyre-life prior only (races strictly before this one)",
                 "history": "the historical prior extended by the ages this race has lap data for",
                 "race": "the oldest age raced in this race"}[edge],
                None if edge == "history_only" else "cliff_history_only")
        else:
            add("cliff_edge", "wear-cliff support edge (as of the snapshot)", _K,
                "the oldest age seen so far as a lower bound, the historical prior from earlier races above it")
        add("cliff_prior", f"tyre-life history prior ({wc.prior_source})", _H, "races strictly before this one")
        add("cliff_shape", f"cliff cost min({wc.kappa} o^2, {wc.cap_s_per_lap}) s/lap past the edge", _P)
    if cfg.compound_proxy.enabled:
        add("proxy_raced", "compound proxy: which compounds count as raced", _W, "decided over the whole race",
            "proxy_off")
        add("proxy_steps", "compound proxy tier-5 steps, weights, sd floor (configs/sim/compound_steps.json)", _H,
            "built from calibration races, some later than the race scored")
        add("proxy_weekend", "compound proxy tiers 2-3 (practice long runs, qualifying)", _WE)
    tr = cfg.traffic.name
    if tr != "none":
        src = cfg.traffic.params.get("source", "recorded")
        own = " at the car's own whole-race pace" if cfg.traffic.params.get("own_pace") and src == "recorded" else ""
        add("traffic", f"traffic ({tr}, source {src})", _W if src == "recorded" else _K,
            ("the recorded field replayed as ghost cars" + own) if src == "recorded" else
            "the other cars projected forward from the rows visible at the snapshot", "traffic_off" if src == "recorded" else None)
    ev = cfg.events.name
    add("events", f"neutralisations (events {ev})",
        {"none": _P, "fixed": _W, "recorded": _W, "hybrid": _W}.get(ev, _H),
        {"none": "none modelled: every remaining lap is green",
         "fixed": "a fixed scenario, the recorded one in replays",
         "recorded": "the race's own SC/VSC laps (and their recorded lap times)",
         "hybrid": "the SC/VSC in progress at the snapshot as recorded (its end, its recorded lap times); later "
                   "ones drawn at the rate model's rates",
         "rate": "EventModel rates (provisional, 2021 Spain) or RaceParams circuit rates"}.get(ev, ""),
        "events_rate" if ev in ("recorded", "hybrid") else None)
    add("event_effects", "SC/VSC lap factors, SC pit-loss factor, lap-1 extra, neutralised tyre ageing "
        "(EventEffects)", _H, "provisional, 2021 Spain")
    add("practice", "practice long runs and qualifying (this weekend)", _WE)
    if cfg.fit_window != "full":
        add("priors", "pre_race base delta, fuel prior, shrinkage taus", _H, "Spain-derived constants")
        cal = cfg.calibrator.name
        from . import calibrators
        learned = calibrators.CALIBRATORS.get(cal).learned or cal == "trained"   # trained: committed artifacts
        add("calibrator", f"calibrator {cal}", _H if learned else _P,
            ("trained on calibration races (committed artifacts; leave-one-out for a calibration race)"
             if cal == "trained" else "trained on other calibration races") if learned else "")
    dp = cfg.degradation.params
    if dp.get("driver_effects", "off") not in ("off", False, None):
        add("driver_effects", f"driver effects ({dp['driver_effects']}) in the degradation fit", wt)
    if cfg.degradation.name.endswith("hinge"):
        add("hinge", f"wear hinge at age {cfg.hinge_knot_default} (knots {cfg.hinge_knots or 'default'})", _P)
    if dp.get("wear_lower_bound") is not None:
        add("wear_lower_bound", f"wear bounded below at {dp['wear_lower_bound']}", _P)
    if dp.get("draw", "none") != "none" or cfg.lap_noise != "none":
        add("draws", f"parameter draws ({dp.get('draw', 'none')}) and lap noise ({cfg.lap_noise})", wt,
            "the fit's own covariance and residual sd")
    add("min_stint_two_compounds", "R1-R3: minimum stint, two dry compounds", _R)
    covered = {"fit_window", "rules", "stint_limit_margin_laps", "stint_limit_rank", "compound_proxy", "traffic",
               "events", "effects", "calibrator", "degradation", "lap_noise", "hinge_knots", "hinge_knot_default",
               "pre_race_base_delta_s", "pre_race_fuel_slope_s_per_lap", "shrinkage", "n_samples", "policy_view",
               "pit_loss", "exclude_snapshot_driver", "wear_cliff", "min_race_rows_to_fit"}
    default = SimConfig().model_dump(mode="json")
    mine = cfg.model_dump(mode="json")
    for k in sorted(mine):
        if k not in covered and mine[k] != default.get(k):
            add(f"field:{k}", f"config field {k} = {json.dumps(mine[k])[:80]}", _U,
                "no ledger rule for this field: declare it with x_ledger")
    by_id = {e["id"]: e for e in out}
    for d in declared or ():
        e = {"role": role, "note": "", "substitute": None, **d, "source": "declared"}
        if e.get("tag") not in TAGS:
            raise ValueError(f"x_ledger {e.get('id')}: tag must be one of {TAGS}")
        by_id[e["id"]] = e
    return list(by_id.values())


def ledger_frame(spec: Spec, data=None) -> pd.DataFrame:
    rows = ledger(spec.world_config(data), "judge", spec.declared.get("judge")) + \
        ledger(spec.belief_config(), "predictive", spec.declared.get("predictive"))
    if spec.world_race:
        for k, v in spec.world_race.items():
            if k == "pit_loss_s":
                rows = [r for r in rows if not (r["role"] == "judge" and r["id"] == "pit_loss")]
                rows.append({"role": "judge", "id": "pit_loss", "input": f"pit loss = {v}",
                             "tag": _H if v == "prior" else _P, "note": "no-hindsight substitute"
                             + (": the pre-race prior from earlier races" if v == "prior" else ""),
                             "substitute": None, "source": "spec"})
                continue
            rows.append({"role": "judge", "id": f"race:{k}", "input": f"RaceParams.{k} = {v}", "tag": _P,
                         "note": "no-hindsight substitute", "substitute": None, "source": "spec"})
    return pd.DataFrame(rows, columns=["role", "id", "input", "tag", "note", "substitute", "source"])


# A judge variant per whole-race input: (world overlay, RaceParams overrides, what replaces it).
SUBSTITUTES: dict[str, dict] = {
    "pre_race_fit": {"world": {"fit_window": "pre_race"}, "race": {},
                     "replaces": "fit_window", "with": "practice long runs plus the pre_race base delta (no race laps)"},
    "pit_loss_prior": {"world": {}, "race": {"pit_loss_s": "prior"}, "replaces": "pit_loss",
                       "with": "the pre-race prior from earlier races (pit_loss.pit_loss_prior: same circuit, else "
                               "across circuits, else 22 s)"},
    "events_rate": {"world": {"events": {"name": "rate", "params": {}}}, "race": {}, "replaces": "events",
                    "with": "SC/VSC drawn at the rate model's rates (no recorded periods or lap times)"},
    "cliff_history_only": {"world": {"wear_cliff": {"world_edge": "history_only"}}, "race": {},
                           "replaces": "cliff_edge",
                           "with": "the wear-cliff edge from the historical prior only, nothing from this race"},
    "r8_off": {"world": {"stint_limit_margin_laps": None}, "race": {}, "replaces": "stint_limit",
               "with": "no stint limit (A4 only: replays do not apply R8)"},
    "proxy_off": {"world": {"compound_proxy": {"enabled": False}}, "race": {}, "replaces": "proxy_raced",
                  "with": "no compound proxy: only raced compounds are priced"},
    "traffic_off": {"world": {"traffic": {"name": "none", "params": {}}}, "race": {}, "replaces": "traffic",
                    "with": "no traffic"},
}
ALL_SUBSTITUTED = "all_substituted"


def substitutes_for(spec: Spec, data=None) -> list[str]:
    led = ledger(spec.world_config(data), "judge", spec.declared.get("judge"))
    names = [e["substitute"] for e in led if e.get("substitute") and e["tag"] == _W]
    if "pit_loss_s" in spec.world_race:
        names = [n for n in names if n != "pit_loss_prior"]
    return [n for n in dict.fromkeys(names) if n in SUBSTITUTES]


def substituted(spec: Spec, names) -> Spec:
    out = spec
    for n in names:
        s = SUBSTITUTES[n]
        out = out.with_factor(out.label, world=s["world"], world_race=s["race"])
    return replace(out, label=f"{spec.label}+{'+'.join(names)}")


# =================================================================================================
# A: judge versus reality
# =================================================================================================
def _actual(g) -> np.ndarray:
    return (g["end_s"] - g["start_s"]).to_numpy(dtype=float)


def _offset(actual, pred, use) -> float:
    use = np.asarray(use, dtype=bool) & np.isfinite(actual) & np.isfinite(pred)
    return _median(actual[use] - pred[use]) if use.sum() >= MIN_OFFSET_LAPS else float("nan")


def _bucket(age) -> str:
    for lo, hi in zip(AGE_BUCKETS[:-1], AGE_BUCKETS[1:]):
        if lo < age <= hi:
            return f"{lo + 1}-{hi}" if hi < 1000 else f"{lo + 1}+"
    return "0"


def _replay(sim, data, race, d):
    """(g, SimResult, predicted lap times) of driver d's real strategy from lap 0 under the
    recorded SC/VSC: as a fixed scenario (calibration.replay_driver), or through the judge's own
    recorded events when it models them (replay_events)."""
    if replay_events(sim.config) == "scenario":
        g, res = calibration.replay_driver(sim, data.frame, race, d)
        return g, res, np.asarray(res.lap_times_s, dtype=float)
    from ..contracts import Snapshot
    frame = data.frame
    g = frame[frame["driver"] == d].sort_values("lap")
    stints = calibration.race_stints(frame, d)
    snap = Snapshot.at_start(race.model_copy(update={"total_laps": len(g)}), d, stints[0][0],
                             tyre_age=stints[0][2]).model_copy(update={"elapsed_s": 0.0})
    res = sim.simulate(snap, Plan.of([[c, k, a] for c, k, a, _ in stints]), rules=calibration.REPLAY_RULES,
                       return_laps=True)
    if not res.valid:
        raise ValueError(f"{d}: replay plan rejected: {res.errors}")
    return g, res, np.asarray(res.lap_times_s, dtype=float)


def _green_pit(g):
    """Rows of in- and out-laps on green (track status 1, 12 or 21), as calibration's pit loss."""
    return g["track_status"].astype(str).isin(["1", "21", "12"]).to_numpy()


def _driver_tables(d, g, res, pred, variant, fit, exclude=None, pit_loss_used=None):
    """Per driver, lap and stint rows of one replay. `exclude` (bool per lap) leaves laps out of the
    pace offset (leave-one-stint-out)."""
    actual = _actual(g)
    kind = calibration._lap_kind(g)
    clean = g["clean"].to_numpy(dtype=bool)
    use = clean if exclude is None else clean & ~exclude
    offset = _offset(actual, pred, use)
    off = 0.0 if not np.isfinite(offset) else offset
    n = len(g)
    real = float(g["end_s"].iloc[-1] - g["start_s"].iloc[0])
    pit = np.isin(kind, ["in_lap", "out_lap"])
    pit_err = float(np.nansum(pred[pit] + off - actual[pit]) + res.pit_loss_total_s)
    neut = np.isin(kind, ["sc", "vsc"])
    stints = calibration.race_stints(g, d)
    stint_id = g["stint"].to_numpy()
    green_stops = 0
    for in_lap in g.loc[g["pit_in"].astype(bool), "lap"]:
        pair = g[g["lap"].isin([in_lap, in_lap + 1])]
        green_stops += int(len(pair) == 2 and _green_pit(pair).all())
    row = {"driver": d, "team": g["team"].iloc[0], "variant": variant, "fit": fit, "laps": n,
           "stops": res.num_stops, "real_s": real, "pred_s": res.race_time_s, "err_s": res.race_time_s - real,
           "pace_offset_s": offset, "err_pace_s": res.race_time_s - real + off * n,
           "pit_err_s": pit_err, "pit_err_per_stop_s": pit_err / res.num_stops if res.num_stops else float("nan"),
           "all_stops_green": bool(res.num_stops and green_stops == res.num_stops),
           "neutral_laps": int(neut.sum()), "neutral_err_s": float(np.nansum(pred[neut] + off - actual[neut])),
           "pit_loss_used_s": pit_loss_used}
    to_end = np.zeros(n, dtype=int)
    for j, (_, m, _, first) in enumerate(stints):
        sel = (g["lap"].to_numpy() >= first) & (g["lap"].to_numpy() < first + m)
        to_end[sel] = first + m - 1 - g["lap"].to_numpy()[sel]
    laps = pd.DataFrame({"driver": d, "variant": variant, "fit": fit, "lap": g["lap"].to_numpy(),
                         "stint": stint_id, "compound": g["compound"].to_numpy(),
                         "tyre_age": g["tyre_age"].to_numpy(dtype=float), "kind": kind, "clean": clean,
                         "laps_to_stint_end": to_end, "resid_s": actual - pred - off})
    srows = []
    for j, (comp, m, a0, first) in enumerate(stints):
        sel = (g["lap"].to_numpy() >= first) & (g["lap"].to_numpy() < first + m) & ~np.isin(
            kind, ["lap1", "in_lap", "out_lap"])
        if exclude is not None and not (exclude & sel).any():
            continue   # leave-one-stint-out reports the held-out stint only
        srows.append({"driver": d, "variant": variant, "fit": fit, "stint": j + 1, "compound": comp,
                      "first_lap": first, "laps": int(sel.sum()), "start_age": a0,
                      "real_s": float(np.nansum(actual[sel])), "pred_s": float(np.nansum(pred[sel] + off)),
                      "err_s": float(np.nansum(pred[sel] + off) - np.nansum(actual[sel]))})
    return row, laps, srows


def _stop_cost(g, pred, offset):
    """[(in_lap, real cost)] of green stops: in+out lap time over the green prediction."""
    actual = _actual(g)
    lapno = g["lap"].to_numpy()
    out = []
    for in_lap in g.loc[g["pit_in"].astype(bool), "lap"]:
        i = np.flatnonzero(lapno == in_lap)
        o = np.flatnonzero(lapno == in_lap + 1)
        if not len(i) or not len(o):
            continue
        pair = g.iloc[[i[0], o[0]]]
        if not _green_pit(pair).all():
            continue
        cost = actual[i[0]] + actual[o[0]] - pred[i[0]] - pred[o[0]] - 2 * (offset if np.isfinite(offset) else 0.0)
        out.append((int(in_lap), float(cost)))
    return out


def _stop_gain_rows(model_for, data, drivers, variant, fit):
    """Predicted against real gain at every real stop of `drivers` (calibration._stops: the last
    two clean laps before, the first two after), the model from model_for(driver)."""
    rows = []
    for d in drivers:
        g = data.frame[data.frame["driver"] == d].sort_values("lap")
        m = model_for(d)
        if m is None:
            continue
        for in_lap, before, after in calibration._stops(g):
            old, new = before["compound"].iloc[-1], after["compound"].iloc[0]
            if not {old, new} <= set(m.compounds):
                continue
            real = float(before["lap_time_s"].mean() - after["lap_time_s"].mean())
            pred = calibration._mean_pred(m, before) - calibration._mean_pred(m, after)
            rows.append({"driver": d, "variant": variant, "fit": fit, "stop": f"{d}:{in_lap}", "in_lap": in_lap,
                         "from": old, "to": new, "real_gain_s": real, "pred_gain_s": pred, "err_s": pred - real,
                         "age_before": float(before["tyre_age"].mean())})
    return rows


def _raced_ages(frame) -> dict:
    return rules.stint_limits(frame, 1)


def _stint_ends(snap, plan) -> list[tuple[str, float]]:
    return [(s.compound, (snap.tyre_age if j == 0 else (s.start_age or 0)) + s.laps) for j, s in enumerate(plan.stints)]


def _beyond(snap, plan, raced, margin) -> dict:
    ends = _stint_ends(snap, plan)
    over = [e - raced[c] if c in raced else math.inf for c, e in ends]
    return {"beyond_raced": bool(max(over) > 0), "beyond_raced_margin": bool(max(over) > (margin or 0)),
            "max_over_raced": float(max(over)) if np.isfinite(max(over)) else float("nan"),
            "unraced_compound": any(c not in raced for c, _ in ends)}


def _reference_row(world, snap, raced, opts, **extra):
    from ..baselines.domain import NoFeasiblePlan
    from .optimizer_bridge import optimize_batch
    # one draw for a judge that draws nothing; its own events' draws (opts.samples) otherwise
    n = 1 if world.config.events.name in ("none", "fixed", "recorded") else max(1, opts.samples or COVERAGE_SAMPLES)
    try:
        res = optimize_batch(snap, world, seed=opts.seed, n_samples=n, max_stops=opts.max_stops, return_times=True)
    except (NoFeasiblePlan, ValueError) as exc:   # no legal plan, or the start compound is not modelled
        return {**extra, "ref_stops": np.nan, "ref_plan": None, "ref_error": f"{type(exc).__name__}: {exc}"}
    plan = res.plan
    return {**extra, "ref_stops": len(plan.stints) - 1, "ref_plan": " ".join(f"{s.compound}:{s.laps}" for s in plan.stints),
            "ref_mean_s": float(res.evaluation.mean_time_s),
            **_beyond(snap, plan, raced, world.config.stint_limit_margin_laps)}


def section_a(data, spec: Spec, opts: Options, variants=("in", "lodo", "loso")) -> dict[str, pd.DataFrame]:
    """A1-A5 for one race: driver, lap, stint, stop-gain and lap-0 reference tables, each row
    tagged variant (in / lodo / loso) and fit (what the tyre parameters saw)."""
    from . import pit_loss as _pl
    cfg_as = spec.world_config(data)
    cfg = _replay_config(cfg_as)
    race = _judge_race(data, spec)
    world = _judge(data, spec, cfg)
    fins = _finishers(data.frame)
    drivers, laps, stints, errors = [], [], [], []
    insample_cost = {}
    for d in fins:
        try:
            g, res, pred = _replay(world, data, race, d)
        except ValueError as exc:
            errors.append({"driver": d, "variant": "in", "error": str(exc)})
            continue
        row, lp, st = _driver_tables(d, g, res, pred, "in", "in-sample", pit_loss_used=world.race_pit_loss_s)
        drivers.append(row)
        laps.append(lp)
        stints += st
        insample_cost[d] = _stop_cost(g, pred, row["pace_offset_s"])
    gains = _stop_gain_rows(lambda d: world.degradation, data, fins, "in", "in-sample")
    if opts.lodo and "lodo" in variants:
        lodo = _judge(data, spec, cfg.model_copy(update={"exclude_snapshot_driver": True}),
                      pit_loss_s=world.race_pit_loss_s)
        # the whole-race pit loss without d's stops: the world's estimand (pit_loss.whole_race, the
        # median green stop against the full fit; the mean under pit_loss.world "replay_mean")
        stops = _pl.stop_table(data.frame)
        green = stops[stops["kind"] == "green"]
        losses = _pl.stop_losses(green, world.degradation, _pl._field_medians(data.frame))
        center = np.mean if cfg.pit_loss.world == "replay_mean" else np.median
        for d in fins:
            if "pit_loss_s" not in spec.world_race:   # a substituted pit loss stays as substituted
                others = losses[(green["driver"] != d).to_numpy()].dropna()
                lodo.race_pit_loss_s = (float(center(others)) if len(others) >= cfg.pit_loss.min_green_stops
                                        else world.race_pit_loss_s)
            try:
                g, res, pred = _replay(lodo, data, race, d)
            except ValueError as exc:
                errors.append({"driver": d, "variant": "lodo", "error": str(exc)})
                continue
            row, lp, st = _driver_tables(d, g, res, pred, "lodo", "leave-one-driver-out",
                                         pit_loss_used=lodo.race_pit_loss_s)
            drivers.append(row)
            laps.append(lp)
            stints += st

        def lodo_model(d):
            return lodo.fitted_for(calibration.snapshot_at(data.frame, race, d, 0)).degradation
        gains += _stop_gain_rows(lodo_model, data, fins, "lodo", "leave-one-driver-out")
    if opts.loso and "loso" in variants:
        for d in fins:
            g0 = data.frame[data.frame["driver"] == d]
            for j in sorted(g0["stint"].unique()):
                held = (data.frame["driver"] == d) & (data.frame["stint"] == j)
                frame = data.frame.assign(clean=data.frame["clean"].astype(bool) & ~held)
                sim = _judge(data, spec, cfg, frame=frame, pit_loss_s=world.race_pit_loss_s)
                try:
                    g, res, pred = _replay(sim, replace(data, frame=frame), race, d)
                except ValueError as exc:
                    errors.append({"driver": d, "variant": "loso", "error": str(exc)})
                    continue
                excl = (g["stint"] == j).to_numpy()
                _, lp, st = _driver_tables(d, g, res, pred, "loso", "leave-one-stint-out", exclude=excl)
                laps.append(lp[excl])
                stints += st
    out = {"a_drivers": pd.DataFrame(drivers), "a_laps": pd.concat(laps, ignore_index=True) if laps else pd.DataFrame(),
           "a_stints": pd.DataFrame(stints), "a_stop_gain": pd.DataFrame(gains),
           "a_errors": pd.DataFrame(errors, columns=["driver", "variant", "error"])}
    # A4: the judge's reference at lap 0 for every field car, against the field's stop counts
    raced = _raced_ages(data.frame)
    field = cross_race.field_stops(data.frame, data.race)
    judge = world if cfg_as == cfg else _judge(data, spec, cfg_as, pit_loss_s=world.race_pit_loss_s)
    refs, seen = [], {}
    for d in field["driver"]:
        snap = calibration.snapshot_at(data.frame, race, d, 0)
        extra = dict(driver=d, k=0, field_stops=int(field.loc[field["driver"] == d, "stops"].iloc[0]))
        # at lap 0 without traffic or pace the reference depends on the start set and race length only
        key = (snap.compound, snap.tyre_age, snap.race.total_laps) if cfg.traffic.name == "none" else d
        if key not in seen:
            seen[key] = _reference_row(judge, snap, raced, opts)
        refs.append({**seen[key], **extra})
    out["a_refs"] = pd.DataFrame(refs)
    out["a_meta"] = pd.DataFrame([{"race_id": data.race_id, "pit_loss_s": world.race_pit_loss_s,
                                   "field_mode": cross_race.stop_mode(field["stops"]),
                                   "world_config": config_key(cfg)}])
    return out


def practice_offsets(data, world) -> pd.Series | None:
    """This weekend's practice pace per driver (pace.practice_pace on the practice fit): known
    before the race, so out of sample for every lap of it."""
    from . import pace
    if data.practice is None or data.practice.empty:
        return None
    try:
        roster = data.frame.groupby("driver")["team"].first()
        t = pace.practice_pace(data.practice, pace.practice_model(world, data.practice), roster=roster)
        return t["offset_s"]
    except Exception:   # practice without a fittable long run
        return None


# =================================================================================================
# Section O: order and time, three versions
# =================================================================================================
def _real_rank(data, fins) -> pd.Series:
    return calibration._real_positions(data.frame, fins, data.results).rank(method="first")


def order_measures(t: pd.DataFrame) -> dict:
    """Order and time measures of one race's predicted totals. `t`: driver, laps, real_s, pred_total_s,
    real_pos. Ranks laps driven first, then time."""
    t = t[np.isfinite(t["pred_total_s"].to_numpy(dtype=float))]
    if len(t) < 3:
        return {}
    laps, pred, real = (t[c].to_numpy(dtype=float) for c in ("laps", "pred_total_s", "real_s"))
    pred_pos = calibration._rank(laps, pred, pred)
    real_pos = t["real_pos"].rank(method="first").to_numpy(dtype=float)
    pos_err = np.abs(pred_pos - real_pos)
    pairs = ok = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if laps[i] == laps[j] and abs(real[i] - real[j]) < CLOSE_PAIR_S and real[i] != real[j]:
                pairs += 1
                ok += int(np.sign(pred[i] - pred[j]) == np.sign(real[i] - real[j]))
    err = pred - real
    same = laps == laps.max()
    rel = err[same] - err[same].mean()
    return {"n_cars": int(len(t)), "close_pairs": pairs, "close_pair_acc": ok / pairs if pairs else float("nan"),
            "spearman": _spearman(pred_pos, real_pos), "mae_pos": _mean(pos_err), "within_1": _share(pos_err <= 1),
            "rel_gap_mae_s": _mae(rel), "abs_mae_s": _mae(err), "bias_s": _mean(err)}


def order_rows_from_a(a_drivers, data, pace_practice, version, label) -> pd.DataFrame:
    """Version (1)/(2) rows: per car the judge's whole-race total with a pace term; `pace` says
    what it was fitted on."""
    rows = []
    if not len(a_drivers) or "variant" not in a_drivers:   # every replay of this variant was rejected
        return pd.DataFrame()
    real_rank = _real_rank(data, list(a_drivers["driver"].unique()))
    for (variant, fit), part in a_drivers[a_drivers["variant"].isin(["in", "lodo"])].groupby(["variant", "fit"]):
        paces = {"insample": part.set_index("driver")["pace_offset_s"].fillna(0.0),
                 "none": pd.Series(0.0, index=part["driver"])}
        if pace_practice is not None:
            paces["practice"] = pace_practice.reindex(part["driver"]).fillna(0.0)
        for pk, p in paces.items():
            t = part.assign(pred_total_s=part["pred_s"].to_numpy() + p.reindex(part["driver"]).to_numpy() * part["laps"].to_numpy(),
                            real_pos=real_rank.reindex(part["driver"]).to_numpy())
            rows.append(t.assign(version=version, label=label, k=0, fit=fit, pace=pk,
                                 in_sample=(fit == "in-sample") or pk == "insample"))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[["version", "label", "k", "fit", "pace", "in_sample", "driver", "laps",
                                               "real_s", "pred_total_s", "real_pos"]]


def predictive_order_rows(data, belief, pace_practice, opts, ks=ORDER_KS) -> pd.DataFrame:
    """Version (3) and B4: every finisher's real remaining strategy from lap k under the belief
    (calibration.midrace: the belief's own fit at k, the recorded SC/VSC laps ahead). Pace is
    as-of (green laps up to k, calibration._pace_offset), practice where that is undefined."""
    ks = [k for k in ks if k < data.race.total_laps - 1]
    g = calibration.midrace(lambda w: belief, data.frame, data.race, laps_k=ks, windows=[belief.config.fit_window],
                            results=data.results, n_samples=max(1, opts.samples))
    if not len(g):
        return pd.DataFrame(), g
    fins = sorted(g["driver"].unique())
    real_rank = _real_rank(data, fins)
    pp = pace_practice.reindex(g["driver"]).to_numpy(dtype=float) if pace_practice is not None else np.full(len(g), np.nan)
    asof = g["pace_offset_s"].to_numpy(dtype=float)
    pace = np.where(np.isfinite(asof), asof, np.where(np.isfinite(pp), pp, 0.0))
    real_s = (g["elapsed_s"] + g["real_remaining_s"]).to_numpy()
    base = g.assign(real_s=real_s, real_pos=real_rank.reindex(g["driver"]).to_numpy(), version="3 predictive",
                    label="predictive", fit=f"as-of ({belief.config.fit_window})", in_sample=False)
    rows = [base.assign(pace="asof", pred_total_s=(g["elapsed_s"] + g["pred_remaining_s"]).to_numpy() + pace * g["horizon"].to_numpy()),
            base.assign(pace="none", pred_total_s=(g["elapsed_s"] + g["pred_remaining_s"]).to_numpy())]
    out = pd.concat(rows, ignore_index=True)[["version", "label", "k", "fit", "pace", "in_sample", "driver", "laps",
                                              "real_s", "pred_total_s", "real_pos"]]
    return out, g


def field_replay_summary(data, cfg) -> pd.DataFrame | None:
    """The whole-field replay of branch feat/traffic-in-eval (src/sim/field_replay.py) when this
    branch has it: its own rows summarised by its own summarize(); None when it is absent."""
    try:
        fr = importlib.import_module("src.sim.field_replay")
    except ImportError:
        return None
    parts = []
    for k in getattr(fr, "KS", ORDER_KS):
        try:
            parts.append(fr.replay(data, cfg, k))
        except Exception as exc:   # an API this tool was not written against: report, do not fail
            return pd.DataFrame([{"race_id": data.race_id, "error": f"{type(exc).__name__}: {exc}"}])
    rows = pd.concat(parts, ignore_index=True)
    return fr.summarize(rows) if hasattr(fr, "summarize") else rows


# =================================================================================================
# B: predictive versus judge
# =================================================================================================
def plan_panel(snap, compounds, real=None, n_one=4, n_two=3) -> list[tuple[str, str, Plan]]:
    """A fixed, policy-independent panel from `snap`: the real remaining strategy, staying out to
    the flag, one stop at n_one laps spread from the soonest legal stop on each compound, and two
    stops at n_two first-stop laps (the second halfway through the rest) on every compound pair.
    Plans that break R1-R3 are left for the simulators' rules to reject."""
    r, mn, cur, age = snap.remaining_laps, snap.race.min_stint_laps, snap.compound, snap.tyre_age
    soonest = max(1, mn - rules.laps_on_set(snap))
    out, seen = [], set()

    def add(name, kind, stints):
        try:
            p = Plan.of(stints)
        except Exception:
            return
        key = tuple((s.compound, s.laps, s.start_age) for s in p.stints)
        if key not in seen:
            seen.add(key)
            out.append((name, kind, p))
    if real is not None:
        add("real", "real", [[s.compound, s.laps, s.start_age] for s in real.stints])
    add("stay", "stay", [[cur, r, age]])
    if r - mn >= soonest:
        for L in sorted(set(np.round(np.linspace(soonest, r - mn, n_one)).astype(int))):
            for c in compounds:
                add(f"1:{L}:{c}", "one", [[cur, int(L), age], [c, int(r - L), 0]])
    if r - 2 * mn >= soonest:
        for L1 in sorted(set(np.round(np.linspace(soonest, r - 2 * mn, n_two)).astype(int))):
            L2 = min(int(L1) + max(mn, (r - int(L1)) // 2), r - mn)
            for c1 in compounds:
                for c2 in compounds:
                    add(f"2:{L1}:{L2}:{c1}:{c2}", "two", [[cur, int(L1), age], [c1, L2 - int(L1), 0], [c2, r - L2, 0]])
    return out


def _legal(sim, snap, plan) -> bool:
    try:
        return not sim.validate(snap, plan)
    except Exception:
        return False


def _factor_coverage(bdeg, wdeg, k) -> list[dict]:
    """Per fitted factor both models carry: the belief's value and sd against the judge's value."""
    rep = getattr(bdeg, "fit_report", None) or {}
    sd = rep.get("sd") or {}
    rows = []
    for c in sorted(set(getattr(bdeg, "coefs", {})) & set(getattr(wdeg, "coefs", {}))):
        for i, name in enumerate(("base_s", "wear_s_per_lap")):
            s = (sd.get(name) or {}).get(c, float("nan"))
            rows.append({"k": k, "factor": f"{name}[{c}]", "belief": float(bdeg.coefs[c][i]),
                         "judge": float(wdeg.coefs[c][i]), "sd": float(s)})
    if hasattr(bdeg, "fuel_slope") and hasattr(wdeg, "fuel_slope"):
        rows.append({"k": k, "factor": "fuel_s_per_lap", "belief": float(bdeg.fuel_slope),
                     "judge": float(wdeg.fuel_slope), "sd": float(sd.get("fuel_s_per_lap", float("nan")))})
    for r in rows:
        z = (r["belief"] - r["judge"]) / r["sd"] if r["sd"] and np.isfinite(r["sd"]) and r["sd"] > 0 else float("nan")
        r.update(z=z, covered_68=abs(z) <= 1 if np.isfinite(z) else np.nan,
                 covered_90=abs(z) <= Z90 if np.isfinite(z) else np.nan)
    return rows


def learned_calibrator(cfg: SimConfig, race_id, pool, profile=None):
    """A fitted learned calibrator for `race_id`, trained on the other races of `pool` (never on
    the race itself); None when cfg's calibrator is not a learned one."""
    from . import calibrators, jobs
    cls = calibrators.CALIBRATORS.get(cfg.calibrator.name)
    if not cls.learned:
        return None
    from .compute import load_profile
    profile = profile or load_profile()
    others = [replace(load(r, profile), config=cfg) for r in pool if canonical_race_id(r) != race_id]
    ds = calibrators.build_dataset(others, laps_k=lambda d: jobs.laps_k_for(d.frame, d.race.total_laps, profile))
    return cls(**cfg.calibrator.params).fit(ds)


def section_b(data, spec: Spec, opts: Options, belief=None, world=None, pool=()) -> dict[str, pd.DataFrame]:
    """B1-B3 on the plan panel at the snapshot grid, the grid references (A4), and the order
    version-3 rows with B4 (the G measure). `belief` / `world` may be prebuilt Simulators."""
    from .agent_tools import BELIEF_SEED_OFFSET
    world = world or _judge(data, spec)
    if belief is None:
        bcfg = spec.belief_config()
        belief = _sim(data, bcfg, calibrator=learned_calibrator(bcfg, data.race_id, pool))
    raced = _raced_ages(data.frame)
    build = getattr(calibration, "snapshot_as_of", calibration.snapshot_at)   # what a policy sees
    n = COVERAGE_SAMPLES if opts.samples == 0 else max(1, opts.samples)
    rows, snaps, refs, factors = [], [], [], []
    fac_done = set()
    for d in _sample_drivers(data, opts.b_drivers, full_distance=True):
        g = data.frame[data.frame["driver"] == d]
        nlaps = len(g)
        for k in range(opts.b_every, nlaps - data.race.min_stint_laps, opts.b_every):
            if not (g["lap"] == k).any():
                continue
            snap = build(data.frame, data.race, d, k)
            # the judge scores the policy's snapshot (its pit loss comes from race_pit_loss_s), with any
            # other RaceParams the spec overrides
            other = {k: v for k, v in spec.world_race.items() if k != "pit_loss_s"}
            snap_w = snap.model_copy(update={"race": snap.race.model_copy(update=other)}) if other else snap
            bfit = belief.fitted_for(snap)
            comps = sorted(set(bfit.modelled) & set(world.fitted_for(snap_w).modelled))
            panel = plan_panel(snap, comps, calibration.remaining_plan(data.frame, d, k), opts.n_one, opts.n_two)
            plans = [p for _, _, p in panel]
            tb = belief.simulate_batch(snap, plans, seed=opts.seed + BELIEF_SEED_OFFSET, n_samples=n, rules=REG_RULES)
            tw = world.simulate_batch(snap_w, plans, seed=opts.seed, n_samples=n, rules=REG_RULES)
            # the same with no SC/VSC on either side: the physics difference without the event model's
            green = EventScenario()
            gb = belief.simulate_batch(snap, plans, seed=opts.seed + BELIEF_SEED_OFFSET, n_samples=n, rules=REG_RULES,
                                       scenario=green)
            gw = world.simulate_batch(snap_w, plans, seed=opts.seed, n_samples=n, rules=REG_RULES, scenario=green)
            phase = "early" if k < nlaps / 3 else "mid" if k < 2 * nlaps / 3 else "late"
            base = {"driver": d, "k": k, "horizon": snap.remaining_laps, "phase": phase}
            ok = tb.valid & tw.valid
            real_i = 0 if panel and panel[0][0] == "real" else None
            for i, (name, kind, p) in enumerate(panel):
                row = {**base, "plan": name, "kind": kind, "stops": len(p.stints) - 1, "pit_now": p.stints[0].laps == 1,
                       "valid": bool(ok[i]), "legal_b": bool(ok[i]) and _legal(belief, snap, p),
                       "legal_w": bool(ok[i]) and _legal(world, snap_w, p),
                       "pred_b_s": float(tb.times[i].mean()) if tb.valid[i] else np.nan,
                       "pred_w_s": float(tw.times[i].mean()) if tw.valid[i] else np.nan}
                row["err_s"] = row["pred_b_s"] - row["pred_w_s"]
                row["pred_b_green_s"] = float(gb.times[i].mean()) if gb.valid[i] else np.nan
                row["pred_w_green_s"] = float(gw.times[i].mean()) if gw.valid[i] else np.nan
                row["err_green_s"] = row["pred_b_green_s"] - row["pred_w_green_s"]
                if n > 1 and ok[i]:
                    q = np.percentile(tb.times[i], [5, 16, 84, 95])
                    j = row["pred_w_s"]
                    row.update(cov_68=bool(q[1] <= j <= q[2]), cov_90=bool(q[0] <= j <= q[3]))
                    if real_i is not None and i != real_i and ok[real_i]:
                        gap_b = tb.times[i] - tb.times[real_i]
                        gq = np.percentile(gap_b, [5, 16, 84, 95])
                        gj = row["pred_w_s"] - float(tw.times[real_i].mean())
                        row.update(gap_cov_68=bool(gq[1] <= gj <= gq[2]), gap_cov_90=bool(gq[0] <= gj <= gq[3]))
                rows.append(row)
            t = pd.DataFrame(rows[-len(panel):])
            snaps.append({**base, **_rank_measures(t)})
            if opts.grid_references:
                refs.append(_reference_row(world, snap_w, raced, opts, driver=d, k=k,
                                           panel_best_w_s=float(t.loc[t["legal_w"], "pred_w_s"].min())
                                           if t["legal_w"].any() else np.nan))
            k0 = getattr(bfit, "window", None)
            if k0 not in fac_done:
                fac_done.add(k0)
                factors += _factor_coverage(bfit.degradation, world.fitted_for(snap_w).degradation, k)
    out = {"b_rows": pd.DataFrame(rows), "b_snaps": pd.DataFrame(snaps), "b_refs": pd.DataFrame(refs),
           "b_factors": pd.DataFrame(factors)}
    pp = practice_offsets(data, world)
    out["o_pred"], out["b_g"] = predictive_order_rows(data, belief, pp, opts)
    return out


def _rank_measures(t: pd.DataFrame) -> dict:
    """B2 at one snapshot over the plans legal under both simulators' own rules."""
    j = t[t["legal_b"] & t["legal_w"]]
    out = {"n_plans": int(len(t)), "n_joint": int(len(j)),
           "fav_b_illegal_w": bool(t["legal_b"].any() and not t.loc[t.loc[t["legal_b"], "pred_b_s"].idxmin(), "legal_w"])
           if t["legal_b"].any() else np.nan}
    if len(j) < 2:
        return out
    fb, fw = j["pred_b_s"].idxmin(), j["pred_w_s"].idxmin()
    out.update(rho=_spearman(j["pred_b_s"], j["pred_w_s"]), regret_s=float(j.loc[fb, "pred_w_s"] - j.loc[fw, "pred_w_s"]),
               same_favourite=bool(fb == fw), fav_b=j.loc[fb, "plan"], fav_w=j.loc[fw, "plan"],
               fav_b_stops=int(j.loc[fb, "stops"]), fav_w_stops=int(j.loc[fw, "stops"]))
    gj = j[np.isfinite(j["pred_b_green_s"]) & np.isfinite(j["pred_w_green_s"])] if "pred_b_green_s" in j else j.iloc[0:0]
    if len(gj) >= 2:
        gb, gw = gj["pred_b_green_s"].idxmin(), gj["pred_w_green_s"].idxmin()
        out.update(rho_green=_spearman(gj["pred_b_green_s"], gj["pred_w_green_s"]),
                   regret_green_s=float(gj.loc[gb, "pred_w_green_s"] - gj.loc[gw, "pred_w_green_s"]))
    p, s = j[j["pit_now"]], j[~j["pit_now"]]
    if len(p) and len(s):
        out["pit_now_agree"] = bool((p["pred_b_s"].min() < s["pred_b_s"].min()) == (p["pred_w_s"].min() < s["pred_w_s"].min()))
        out["pit_now_w"] = bool(p["pred_w_s"].min() < s["pred_w_s"].min())
    return out


def future_blindness(spec: Spec) -> dict:
    """B5: tests/future_blindness.py (branch test/future-blindness) when present, run on its
    synthetic race with this belief config, every known leak neutralised (new leaks only), then
    with none neutralised. "not checked" when the harness is absent."""
    path = ROOT / "tests" / "future_blindness.py"
    if not path.exists():
        return {"status": "not checked", "detail": "tests/future_blindness.py is not on this branch"}
    import tempfile
    from unittest import mock
    import yaml
    sys.path.insert(0, str(path.parent))
    try:
        fb = importlib.import_module("future_blindness")
        from ..eval import score
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "belief.yaml"
            cfg.write_text(yaml.safe_dump(spec.belief_config().model_dump(mode="json")))
            with mock.patch.object(score, "BELIEF_CONFIG", cfg):
                inp = fb.synthetic_input()
                orig = fb.build(inp)
                drivers = sorted(orig.frame["driver"].unique())
                points = [(drivers[0], 5), (drivers[-1], 30)]
                new = fb.check(inp, orig, points, "scramble", neutralise=tuple(fb.KNOWN_LEAKS))
                allp = fb.check(inp, orig, points, "scramble", neutralise=())
        return {"status": "pass" if not new else "fail", "new_leak_differences": len(new),
                "differences_with_known_leaks": len(allp), "known_leaks": sorted(fb.KNOWN_LEAKS),
                "fixed_leaks": sorted(getattr(fb, "FIXED_LEAKS", {})),
                "detail": "; ".join(map(str, new[:5]))}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.path.remove(str(path.parent))


# =================================================================================================
# D: component targets
# =================================================================================================
class NotPresent(Exception):
    """A component's code is not on this branch."""


@dataclass
class Ctx:
    data: object
    world: Simulator
    belief: Simulator
    ks: list
    ref_driver: str
    _cache: dict = field(default_factory=dict)

    def snap(self, k):
        build = getattr(calibration, "snapshot_as_of", calibration.snapshot_at)
        return build(self.data.frame, self.data.race, self.ref_driver, k)

    def belief_at(self, k):
        return self.belief.fitted_for(self.snap(k))

    def world_run(self):
        if "run" not in self._cache:
            self._cache["run"] = calibration.run(self.world, self.data.frame, self.data.race)
        return self._cache["run"]


@dataclass
class Component:
    """`predict(ctx)` -> rows {key, source, pred, sd?} from allowed information (source names what it
    saw); `target(ctx)` -> rows {key, target_kind, target} measured on the whole race. Rows join on
    key. Raise NotPresent when the code is absent on this branch."""
    name: str
    predict: Callable
    target: Callable
    note: str = ""


COMPONENTS: dict[str, Component] = {}


def register(c: Component) -> Component:
    COMPONENTS[c.name] = c
    return c


def _tyre_history():
    """src/sim/tyre_history.py (tyre_life_prior, observed_tyre_life), None where the branch lacks it."""
    try:
        return importlib.import_module("src.sim.tyre_history")
    except ImportError:
        return None


def _tyre_life_predict(ctx):
    rows = []
    th = _tyre_history()
    if th is not None:
        for role in th.ROLES:
            try:
                pr = th.tyre_life_prior(ctx.data.race_id, role)
            except Exception as exc:   # a race the history table cannot place: reported, not fatal
                rows.append({"key": role, "source": f"history prior (tyre_life_prior): {type(exc).__name__}",
                             "pred": np.nan})
                continue
            rows.append({"key": role, "source": "history prior (tyre_life_prior)", "pred": float(pr.edge_laps),
                         "sd": float(pr.sd_laps)})
    else:
        rows.append({"key": "*", "source": "history prior (tyre_life_prior)", "pred": np.nan, "not_present": True})
    s0 = ctx.snap(0)
    if hasattr(ctx.world, "cliffs_for"):   # the judge's wear-cliff edge (the scorer's, may read this race)
        rows += [{"key": c, "source": f"judge cliff edge ({ctx.world.config.wear_cliff.world_edge})", "pred": float(cl.age)}
                 for c, cl in ctx.world.cliffs_for(s0).items()]
    for k in ctx.ks:
        lim = getattr(ctx.belief_at(k), "stint_limits", None) or {}
        rows += [{"key": c, "source": f"belief R8 limit @k{k}", "pred": float(v)} for c, v in lim.items()]
        if hasattr(ctx.belief, "cliffs_for"):   # the belief's as-of edge: max(seen so far, prior)
            for c, cl in ctx.belief.cliffs_for(ctx.snap(k)).items():
                point = max(cl.lower, cl.prior) if cl.prior is not None else cl.age
                rows.append({"key": c, "source": f"belief cliff edge @k{k}", "pred": float(point),
                             "sd": float(cl.prior_sd) if cl.prior is not None and cl.prior > cl.lower else np.nan})
        f = ctx.data.frame
        seen = f[f["lap"] <= k].groupby("compound")["tyre_age"].max()
        prac = ctx.data.practice.groupby("compound")["tyre_age"].max() if ctx.data.practice is not None else pd.Series(dtype=float)
        for c in sorted(set(seen.index) | set(prac.index)):
            rows.append({"key": c, "source": f"as-of data (race to k{k}, practice)",
                         "pred": float(np.nanmax([seen.get(c, np.nan), prac.get(c, np.nan)]))})
    return rows


def _tyre_life_target(ctx):
    f = ctx.data.frame
    rows = [{"key": c, "target_kind": "oldest age run", "target": float(v)}
            for c, v in f.groupby("compound")["tyre_age"].max().items()]
    ends = f.groupby(["driver", "stint"]).agg(compound=("compound", "first"), age=("tyre_age", "max"))
    for c, part in ends.groupby("compound"):
        rows.append({"key": c, "target_kind": "median stint-end age", "target": _median(part["age"])})
    th = _tyre_history()
    if th is not None:
        try:   # the history's own whole-race observation: the quantile of usable stint-end ages
            obs = th.observed_tyre_life(f, race_id=ctx.data.race_id)
            rows += [{"key": str(r["role"]), "target_kind": "observed edge (tyre_history.observed_tyre_life)",
                      "target": float(r["edge_laps"])} for _, r in obs.iterrows()]
        except Exception:   # the frame lacks a column the observation needs: the two above stand
            pass
    return rows


register(Component("tyre_life", _tyre_life_predict, _tyre_life_target,
                   "support edge / fall-off age per compound: the history prior (tyre_history.tyre_life_prior), the "
                   "belief's R8 edge and the as-of data edge, against the whole race"))


def _within_stint(ctx):
    fuel = float(getattr(ctx.world.degradation, "fuel_slope", 0.0) or 0.0)
    w = cross_race.wear_check(ctx.data.frame, ctx.world.degradation)
    return [{"key": r.compound, "target_kind": "within-stint slope less fuel", "target": float(r.within_stint_wear_s_per_lap)}
            for r in w.itertuples()], fuel


def _wear_predict(ctx):
    rows = []
    for k in ctx.ks:
        m = ctx.belief_at(k).degradation
        sd = ((getattr(m, "fit_report", None) or {}).get("sd") or {}).get("wear_s_per_lap") or {}
        rows += [{"key": c, "source": f"belief @k{k}", "pred": float(v[1]), "sd": float(sd.get(c, np.nan))}
                 for c, v in m.coefs.items()]
    rows += [{"key": c, "source": "judge (full fit)", "pred": float(v[1])} for c, v in ctx.world.degradation.coefs.items()]
    return rows


register(Component("wear_rate", _wear_predict, lambda ctx: _within_stint(ctx)[0],
                   "wear per compound against the race's within-stint slopes (late-stint shape: Section A lap residuals)"))


def _gain_predict(ctx):
    rows = []
    fins = _finishers(ctx.data.frame)
    for k in ctx.ks:
        m = ctx.belief_at(k).degradation
        for r in _stop_gain_rows(lambda d: m, ctx.data, fins, "", ""):
            if r["in_lap"] > k:
                rows.append({"key": r["stop"], "source": f"belief @k{k}", "pred": r["pred_gain_s"]})
    return rows


def _gain_target(ctx):
    return [{"key": r["stop"], "target_kind": "real gain at the stop", "target": r["real_gain_s"]}
            for r in _stop_gain_rows(lambda d: ctx.world.degradation, ctx.data, _finishers(ctx.data.frame), "", "")]


register(Component("fresh_tyre_gain", _gain_predict, _gain_target,
                   "lap-time gain at real stops: last two clean laps before less first two after"))


PAIRS = (("SOFT", "MEDIUM"), ("MEDIUM", "HARD"), ("SOFT", "HARD"))


def _fresh_gap(m, mid):
    out = {}
    for a, b in PAIRS:
        if {a, b} <= set(m.compounds):
            ta = float(m.lap_times(a, np.array([1.0]), np.array([float(mid)]))[0])
            tb = float(m.lap_times(b, np.array([1.0]), np.array([float(mid)]))[0])
            out[f"{a}-{b}"] = ta - tb
    return out


def _gap_predict(ctx):
    mid = ctx.data.race.total_laps // 2
    rows = []
    for k in ctx.ks:
        rows += [{"key": p, "source": f"belief @k{k}", "pred": v} for p, v in _fresh_gap(ctx.belief_at(k).degradation, mid).items()]
    return rows


def _gap_target(ctx):
    from . import proxy
    mid = ctx.data.race.total_laps // 2
    rows = [{"key": p, "target_kind": "race (judge full fit, fresh tyre, mid-race)", "target": v}
            for p, v in _fresh_gap(ctx.world.degradation, mid).items()
            if not any(c in proxy.proxied(ctx.world.degradation.fit_report) for c in p.split("-"))]
    q = proxy.qualifying_gaps(getattr(ctx.data, "qualifying", None))
    if q is not None:
        for a, b in PAIRS:
            if a in q["gap"] and b in q["gap"]:
                rows.append({"key": f"{a}-{b}", "target_kind": "qualifying", "target": float(q["gap"][a] - q["gap"][b])})
    return rows


register(Component("compound_gap", _gap_predict, _gap_target, "fresh-tyre pace gap between compounds"))


def _fuel_predict(ctx):
    rows = [{"key": "fuel", "source": "prior (config)", "pred": float(ctx.belief.config.pre_race_fuel_slope_s_per_lap)}]
    for k in ctx.ks:
        m = ctx.belief_at(k).degradation
        sd = ((getattr(m, "fit_report", None) or {}).get("sd") or {}).get("fuel_s_per_lap", np.nan)
        rows.append({"key": "fuel", "source": f"belief @k{k}", "pred": float(getattr(m, "fuel_slope", np.nan)),
                     "sd": float(sd)})
    return rows


register(Component("fuel", _fuel_predict,
                   lambda ctx: [{"key": "fuel", "target_kind": "race (judge full fit)",
                                 "target": float(getattr(ctx.world.degradation, "fuel_slope", np.nan))}],
                   "fuel/track-evolution slope s per lap"))


def _charged_pit(sim, snap, sc=False, n=20):
    """What `sim` charges for one stop at the end of lap k + 1 (green, or under SC), the mean over
    `n` draws (an as-of pit loss is drawn per sample with its sd)."""
    other = next((c for c in sorted(sim.fitted_for(snap).modelled) if c != snap.compound), snap.compound)
    plan = Plan.of([[snap.compound, 1, snap.tyre_age], [other, snap.remaining_laps - 1, 0]])
    scen = EventScenario(sc_laps=[snap.laps_completed + 1] if sc else [])
    b = sim.simulate_batch(snap, [plan], scenario=scen, rules=["lap_sum", "known_compounds"], n_samples=n)
    return float(b.pit_loss[0].mean()) if b.valid[0] else np.nan


def _pit_module():
    try:
        return importlib.import_module("src.sim.pit_loss")
    except ImportError:
        raise NotPresent("src/sim/pit_loss.py is not on this branch")


def _pit_predict(ctx):
    pl = _pit_module()
    cfg = ctx.belief.config.pit_loss
    pr = pl.pit_loss_prior(ctx.data.race_id, cfg)
    rows = [{"key": "green", "source": f"prior from earlier races ({pr['source']})", "pred": pr["mean_s"], "sd": pr["sd_s"]},
            {"key": "green", "source": "constant (calibrate.DEFAULT_PIT_LOSS_S)", "pred": calibrate.DEFAULT_PIT_LOSS_S}]
    for k in ctx.ks:
        s = ctx.snap(k)
        if s.remaining_laps < 2:
            continue
        est = pl.estimate(ctx.data.frame, ctx.data.race_id, ctx.ref_driver, k, ctx.belief_at(k).degradation, cfg)
        rows.append({"key": "green", "source": f"as-of estimate @k{k} (pit_loss.estimate)", "pred": est["pit_loss_s"],
                     "sd": est["sd_s"]})
        rows.append({"key": "sc", "source": f"as-of estimate @k{k} (pit_loss.estimate)",
                     "pred": est["pit_loss_s"] * est["sc_factor"], "sd": est["pit_loss_s"] * est["sc_factor_sd"]})
        rows.append({"key": "green", "source": f"belief charged @k{k}", "pred": _charged_pit(ctx.belief, s)})
        rows.append({"key": "sc", "source": f"belief charged @k{k}", "pred": _charged_pit(ctx.belief, s, sc=True)})
    return rows


def _pit_target(ctx):
    from .config import PitLossModel
    pl = _pit_module()
    model = ctx.world.degradation
    stops = pl.stop_table(ctx.data.frame)
    med = pl._field_medians(ctx.data.frame)
    whole = pl.whole_race(ctx.data.frame, model, ctx.data.race_id, PitLossModel())
    rows = [{"key": "green", "target_kind": "whole race, robust (pit_loss.whole_race: median green stop)",
             "target": float(whole["pit_loss_s"])},
            {"key": "green", "target_kind": "whole race, replay mean (calibrate._pit_loss)",
             "target": float(ctx.data.race.pit_loss_s)}]
    sc = pl.stop_losses(stops[stops["kind"] == "sc"], model, med).dropna()
    if len(sc):
        rows.append({"key": "sc", "target_kind": "whole race, median SC stop", "target": float(sc.median())})
    return rows


register(Component("pit_loss", _pit_predict, _pit_target,
                   "time lost to a stop: the prior from earlier races and the as-of estimate (pit_loss.py) against the "
                   "world's whole-race value, green and under SC"))


def _lap1_predict(ctx):
    return [{"key": "lap1_extra", "source": "prior (config)", "pred": float(ctx.belief.config.effects.lap1_extra_s)},
            {"key": "out_lap_warmup", "source": "model (no warm-up term: 0)", "pred": 0.0}]


def _lap1_target(ctx):
    table, summary, _ = ctx.world_run()
    res = []
    for d in _finishers(ctx.data.frame):
        g, r, pred = _replay(ctx.world, ctx.data, ctx.data.race, d)
        off = float(table.loc[d, "pace_offset_s"])
        actual = _actual(g)
        out_laps = np.flatnonzero(g["pit_out"].to_numpy(dtype=bool))
        for i in out_laps:
            if i + 1 < len(g) and str(g["track_status"].iloc[i + 1]) == "1":
                res.append(actual[i + 1] - pred[i + 1] - off)
    return [{"key": "lap1_extra", "target_kind": "observed (median lap-1 residual + prior)",
             "target": float(summary["observed_lap1_extra_s"])},
            {"key": "out_lap_warmup", "target_kind": "median residual of the first lap after the out-lap", "target": _median(res)}]


register(Component("lap1_and_warmup", _lap1_predict, _lap1_target, "standing-start extra and first-lap-of-stint penalty"))


def _traffic_predict(ctx):
    rows = []
    for name, sim in (("judge", ctx.world), ("belief", ctx.belief)):
        try:
            rj = calibration.rejoin(sim, ctx.data.frame, ctx.data.race)
        except ValueError:
            continue
        rows += [{"key": f"{r.driver}:{r.in_lap}", "source": f"{name} rejoin position (horizon 2)", "pred": float(r.pred_pos)}
                 for r in rj.itertuples()]
        try:
            ta = importlib.import_module("src.sim.traffic_accuracy")
            lr = ta.long_rejoin(sim, ctx.data.frame, ctx.data.race)
            rows += [{"key": f"{r['driver']}:{r['in_lap']}:5", "source": f"{name} rejoin position (horizon 5, traffic_accuracy)",
                      "pred": float(r["pred_pos"])} for _, r in lr.iterrows()]
        except ImportError:
            pass
        except Exception:   # the branch's API moved: the horizon-2 rows stand
            pass
    return rows


def _traffic_target(ctx):
    rj = calibration.rejoin(ctx.world, ctx.data.frame, ctx.data.race)
    rows = [{"key": f"{r.driver}:{r.in_lap}", "target_kind": "real position after the out-lap", "target": float(r.real_pos)}
            for r in rj.itertuples()]
    try:
        ta = importlib.import_module("src.sim.traffic_accuracy")
        lr = ta.long_rejoin(ctx.world, ctx.data.frame, ctx.data.race)
        rows += [{"key": f"{r['driver']}:{r['in_lap']}:5", "target_kind": "real position three laps later",
                  "target": float(r["real_pos"])} for _, r in lr.iterrows()]
    except Exception:
        pass
    return rows


register(Component("traffic_rejoin", _traffic_predict, _traffic_target,
                   "rejoin position after a real stop (traffic_accuracy's horizon 5 when branch feat/traffic-in-eval is present)"))


def _neutral_predict(ctx):
    from .config import EventModel
    rows = []
    for name, sim in (("judge", ctx.world), ("belief", ctx.belief)):
        ev = sim.config.events
        if ev.name == "none":
            vals = {"sc_periods": 0.0, "vsc_periods": 0.0, "sc_laps": 0.0, "vsc_laps": 0.0}
        elif ev.name == "recorded":   # the race's own periods: the hindsight target itself
            _, summary, _ = ctx.world_run()
            vals = {"sc_periods": float(summary["sc_count"]), "vsc_periods": float(summary["vsc_count"]),
                    "sc_laps": float(summary["sc_laps"]), "vsc_laps": float(summary["vsc_laps"])}
        elif ev.name in ("rate", "hybrid"):   # hybrid from lap 0: every period drawn at the rates
            em = EventModel(**{k: v for k, v in ev.params.items() if k in EventModel.model_fields})
            r = ctx.data.race
            p_sc = r.sc_rate_per_lap if r.sc_rate_per_lap is not None else em.p_sc
            d_sc = r.sc_mean_laps if r.sc_mean_laps is not None else em.d_sc
            p_v = r.vsc_rate_per_lap if r.vsc_rate_per_lap is not None else em.p_vsc
            d_v = r.vsc_mean_laps if r.vsc_mean_laps is not None else em.d_vsc
            n = r.total_laps
            vals = {"sc_periods": p_sc * n, "vsc_periods": p_v * n, "sc_laps": p_sc * n * d_sc, "vsc_laps": p_v * n * d_v}
        else:
            continue
        rows += [{"key": k, "source": f"{name} expected ({ev.name})", "pred": v} for k, v in vals.items()]
        rows += [{"key": "sc_lap_factor", "source": f"{name} config", "pred": float(sim.config.effects.sc_lap_factor)},
                 {"key": "vsc_lap_factor", "source": f"{name} config", "pred": float(sim.config.effects.vsc_lap_factor)}]
    return rows


def _neutral_target(ctx):
    _, summary, _ = ctx.world_run()
    rows = [{"key": f"{k}", "target_kind": "occurred", "target": float(summary[k])}
            for k in ("sc_laps", "vsc_laps")]
    rows += [{"key": "sc_periods", "target_kind": "occurred", "target": float(summary["sc_count"])},
             {"key": "vsc_periods", "target_kind": "occurred", "target": float(summary["vsc_count"])},
             {"key": "sc_lap_factor", "target_kind": "observed (mean SC lap / ref lap)", "target": float(summary["observed_sc_factor"])},
             {"key": "vsc_lap_factor", "target_kind": "observed (median VSC lap / green prediction)",
              "target": float(summary["observed_vsc_factor"])}]
    return rows


register(Component("neutralisations", _neutral_predict, _neutral_target,
                   "event rate (expected periods and laps over the race) and lap time under SC/VSC"))


def _neutral_lap_rows(ctx, which):
    """Per neutralised kind, the replays' mean predicted (which "pred") or real ("real") lap time
    on the cars' SC and VSC laps (pit laps left out), judge and belief alike."""
    key = f"neutral_laps_{which}"
    if key not in ctx._cache:
        out = {}
        for name, sim in (("judge", ctx.world), ("belief", ctx.belief)):
            pred, real = {"sc": [], "vsc": []}, {"sc": [], "vsc": []}
            for d in _finishers(ctx.data.frame):
                try:
                    g, _, p = _replay(sim, ctx.data, ctx.data.race, d)
                except ValueError:
                    continue
                kind = calibration._lap_kind(g)
                for kd in ("sc", "vsc"):
                    m = kind == kd
                    pred[kd] += list(p[m])
                    real[kd] += list(_actual(g)[m])
            out[name] = (pred, real)
        ctx._cache["neutral_laps_pred"] = {n: v[0] for n, v in out.items()}
        ctx._cache["neutral_laps_real"] = out["judge"][1]
    return ctx._cache[key]


def _neutral_laps_predict(ctx):
    rows = []
    for name, pred in _neutral_lap_rows(ctx, "pred").items():
        sim = ctx.world if name == "judge" else ctx.belief
        how = "recorded lap times" if (sim.config.events.name in ("recorded", "hybrid") and name == "judge"
                                       and sim.config.events.params.get("lap_times", "recorded") == "recorded") else "factors"
        rows += [{"key": kd, "source": f"{name} replay ({how})", "pred": _mean(v)} for kd, v in pred.items() if v]
    return rows


def _neutral_laps_target(ctx):
    return [{"key": kd, "target_kind": "mean real lap time on the cars' neutralised laps", "target": _mean(v)}
            for kd, v in _neutral_lap_rows(ctx, "real").items() if v]


register(Component("neutralised_laps", _neutral_laps_predict, _neutral_laps_target,
                   "lap time under SC and VSC in whole-race replays (recorded lap times in a recorded/hybrid judge)"))


def section_d(data, spec: Spec, opts: Options, world=None, belief=None, pool=()) -> dict[str, pd.DataFrame]:
    world = world or _judge(data, spec, _replay_config(spec.world_config(data)))
    if belief is None:
        bcfg = spec.belief_config()
        belief = _sim(data, bcfg, calibrator=learned_calibrator(bcfg, data.race_id, pool))
    fins = _finishers(data.frame)
    ks = [k for k in (0, 10, 20, data.race.total_laps // 2) if k < data.race.total_laps - 5]
    ctx = Ctx(data, world, belief, sorted(set(ks)), fins[0])
    rows, status = [], []
    for name, comp in COMPONENTS.items():
        try:
            pred = pd.DataFrame(comp.predict(ctx))
            targ = pd.DataFrame(comp.target(ctx))
        except NotPresent as exc:
            status.append({"component": name, "status": "not present", "detail": str(exc)})
            continue
        except Exception as exc:
            status.append({"component": name, "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        missing = pred[pred["not_present"].eq(True)] if "not_present" in pred else pred.iloc[0:0]
        for src in missing["source"].unique() if len(missing) else []:
            status.append({"component": name, "status": "not present", "detail": f"predictor {src}"})
        pred = pred.drop(index=missing.index)
        if not len(pred) or not len(targ):
            status.append({"component": name, "status": "no rows", "detail": ""})
            continue
        if "sd" not in pred:
            pred["sd"] = np.nan
        j = pred.merge(targ, on="key")
        j = j.assign(component=name, err=j["pred"] - j["target"])
        z = j["err"] / j["sd"].where(j["sd"] > 0)
        j["covered_68"] = np.where(z.notna(), z.abs() <= 1, np.nan)
        j["covered_90"] = np.where(z.notna(), z.abs() <= Z90, np.nan)
        rows.append(j[["component", "key", "source", "target_kind", "pred", "sd", "target", "err", "covered_68", "covered_90"]])
        status.append({"component": name, "status": "ok", "detail": comp.note})
    return {"d_rows": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), "d_status": pd.DataFrame(status)}


# =================================================================================================
# running one race, many races
# =================================================================================================
def race_tables(race_id: str, spec: Spec, opts: Options, cache_dir=None, pool=(), profile=None) -> dict[str, pd.DataFrame]:
    """Every section's tables for one race under `spec`, cached per section."""
    data = load(race_id, profile)
    wcfg, bcfg = spec.world_config(data), spec.belief_config()
    wkey = config_key(wcfg) + _sha12(json.dumps(spec.world_race, sort_keys=True))
    bkey = config_key(bcfg) + (_sha12(json.dumps(sorted(pool))) if _is_learned(bcfg) else "")
    out = dict(_cached(cache_dir, race_id, "A", (wkey, opts.key("A")), lambda: section_a(data, spec, opts)))
    out.update(_cached(cache_dir, race_id, "B", (wkey, bkey, opts.key("B")), lambda: section_b(data, spec, opts, pool=pool)))
    if opts.components:
        out.update(_cached(cache_dir, race_id, "D", (wkey, bkey, opts.key("D")), lambda: section_d(data, spec, opts, pool=pool)))
    if opts.hindsight_variants:
        hv = []
        subs = substitutes_for(spec, data)
        sets = [[s] for s in subs] + ([subs] if len(subs) > 1 else [])
        for names in sets:
            v = substituted(spec, names)
            vkey = config_key(v.world_config(data)) + _sha12(json.dumps(v.world_race, sort_keys=True))
            label = names[0] if len(names) == 1 else ALL_SUBSTITUTED

            def run_variant(v=v):
                try:
                    return section_a(data, v, opts, variants=("in", "lodo"))
                except Exception as exc:
                    return {"error": pd.DataFrame([{"error": f"{type(exc).__name__}: {exc}"}])}
            t = _cached(cache_dir, race_id, "H", (vkey, opts.key("H")), run_variant)
            hv.append((label, names, {k: v.assign(race_id=race_id) for k, v in t.items()}))
        out["h_variants"] = hv
    pp = practice_offsets(data, _sim(data, wcfg))
    order = [order_rows_from_a(out["a_drivers"], data, pp, "1 judge", "as configured")]
    for label, names, t in out.get("h_variants", []):
        if "a_drivers" in t:
            order.append(order_rows_from_a(t["a_drivers"], data, pp, "2 judge, substituted", label))
    if len(out.get("o_pred", [])):
        order.append(out["o_pred"])
    out["o_rows"] = pd.concat([o for o in order if len(o)], ignore_index=True)
    fr = field_replay_summary(data, wcfg) if opts.field_replay and importlib.util.find_spec("src.sim.field_replay") else None
    if fr is not None:
        out["o_field_replay"] = fr
    out["ledger"] = ledger_frame(spec, data)
    return {k: (v.assign(race_id=race_id) if isinstance(v, pd.DataFrame) else v) for k, v in out.items()}


def _is_learned(cfg) -> bool:
    from . import calibrators
    return bool(calibrators.CALIBRATORS.get(cfg.calibrator.name).learned)


def _job(item):
    race_id, spec, opts, cache_dir, pool = item
    import warnings
    warnings.simplefilter("ignore", category=FutureWarning)
    t0 = time.time()
    try:
        return race_id, spec.label, race_tables(race_id, spec, opts, cache_dir, pool), time.time() - t0, None
    except Exception:
        return race_id, spec.label, None, time.time() - t0, traceback.format_exc()


def run_specs(specs: list[Spec], races, opts: Options, cache_dir, n_jobs=None, log=print, failures=None) -> dict:
    """{label: {race_id: tables}} for every (spec, race), in a process pool. A (spec, race) that
    fails (a race that does not load, a configuration the race cannot run) is reported in the
    log and in `failures` ([{label, race_id, error}]) and left out, never fatal."""
    races = guard(races)
    items = [(r, s, opts, str(cache_dir) if cache_dir else None, races) for s in specs for r in races]
    n_jobs = n_jobs or min(len(items), max(1, (os.cpu_count() or 2) // 2))
    out = {s.label: {} for s in specs}
    if n_jobs > 1 and len(items) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            done = ex.map(_job, items)
            results = list(done)
    else:
        results = [_job(i) for i in items]
    for race_id, label, tables, secs, err in results:
        if err:
            last = err.strip().splitlines()[-1]
            log(f"  {label:<28} {race_id:<26} FAILED, skipped: {last}")
            if failures is not None:
                failures.append({"label": label, "race_id": race_id, "error": last})
            continue
        log(f"  {label:<28} {race_id:<26} {secs:6.1f} s")
        out[label][race_id] = tables
    return out


def concat(per_race: dict, key: str) -> pd.DataFrame:
    parts = [t[key] for t in per_race.values() if key in t and isinstance(t[key], pd.DataFrame) and len(t[key])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =================================================================================================
# summaries
# =================================================================================================
O_KEYS = ["version", "label", "k", "fit", "pace", "in_sample"]
ORDER_COLS = ["close_pair_acc", "close_pairs", "spearman", "mae_pos", "within_1", "rel_gap_mae_s", "abs_mae_s", "bias_s", "n_cars"]


def order_table(o_rows: pd.DataFrame) -> pd.DataFrame:
    """Order and time measures per (version, label, k, fit, pace) and race, plus "pooled": close
    pairs pooled over races, the rest the mean of the race values."""
    if not len(o_rows):
        return pd.DataFrame()
    rows = []
    for key, part in o_rows.groupby(O_KEYS + ["race_id"], sort=False, dropna=False):
        rows.append({**dict(zip(O_KEYS + ["race_id"], key)), **order_measures(part)})
    t = pd.DataFrame(rows)
    pooled = []
    for key, part in t.groupby(O_KEYS, sort=False, dropna=False):
        pairs = part["close_pairs"].sum()
        pooled.append({**dict(zip(O_KEYS, key)), "race_id": "pooled",
                       "close_pair_acc": float((part["close_pair_acc"] * part["close_pairs"]).sum() / pairs) if pairs else np.nan,
                       "close_pairs": int(pairs), **{c: _mean(part[c]) for c in ORDER_COLS if c not in ("close_pair_acc", "close_pairs")}})
    return pd.concat([t, pd.DataFrame(pooled)], ignore_index=True)[O_KEYS + ["race_id"] + ORDER_COLS]


# --- the noise floor ------------------------------------------------------------------------------
# Levels of hindsight handed to the judge's whole-race replay (version 1, the headline's
# leave-one-driver-out fit and whole-race pace offset), each lap's error zeroed or corrected
# where the level knows the real value. What the last level leaves is incidents, driver errors,
# lap-to-lap noise around each stint's pace and unmodelled effects: no simulator of these inputs
# can order the field better than it does.
CEILING_PARTS = {
    "stint_pace": "each stint's own pace offset (mean clean-lap residual of that stint, five or more clean laps; else the car's), on every lap of the stint",
    "pit_laps": "in- and out-laps as recorded, no charged pit loss (the real stop durations and pit-lane driving)",
    "neutral_laps": "SC and VSC laps as recorded",
    "lap1": "lap 1 as recorded (the start)",
}
CEILING_LEVELS = [("0 judge", ()), ("1 +stint pace", ("stint_pace",)),
                  ("2 +pit laps", ("stint_pace", "pit_laps")),
                  ("3 +neutral laps", ("stint_pace", "pit_laps", "neutral_laps")),
                  ("4 +lap 1", ("stint_pace", "pit_laps", "neutral_laps", "lap1")),
                  ("alone: pit laps", ("pit_laps",)), ("alone: neutral laps", ("neutral_laps",)),
                  ("alone: lap 1", ("lap1",))]


def ceiling_errors(laps: pd.DataFrame, drivers: pd.DataFrame, parts) -> pd.Series:
    """Per driver, the replay's whole-race time error (simulated less real, pace term included)
    when the judge is handed `parts` (CEILING_PARTS) of the real race. `laps` / `drivers` are one
    race's a_laps / a_drivers of one variant. With no parts it is the headline's error
    (err_pace_s): the charged pit loss less the summed lap residuals."""
    out = {}
    for d, g in laps.groupby("driver"):
        row = drivers[drivers["driver"] == d]
        if not len(row):
            continue
        resid = g["resid_s"].to_numpy(dtype=float)
        kind = g["kind"].to_numpy()
        pit = np.isin(kind, ["in_lap", "out_lap"])
        charged = float(row["pit_err_s"].iloc[0] + np.nansum(resid[pit]))   # the pit loss the replay charged
        e = -resid
        if "stint_pace" in parts:
            clean = g["clean"].to_numpy(dtype=bool) & np.isfinite(resid)
            for j in np.unique(g["stint"].to_numpy()):
                sel = g["stint"].to_numpy() == j
                if (sel & clean).sum() >= MIN_OFFSET_LAPS:
                    e[sel] += float(np.mean(resid[sel & clean]))
        if "pit_laps" in parts:
            e[pit] = 0.0
            charged = 0.0
        if "neutral_laps" in parts:
            e[np.isin(kind, ["sc", "vsc"])] = 0.0
        if "lap1" in parts:
            e[kind == "lap1"] = 0.0
        out[d] = float(np.nansum(e) + charged)
    return pd.Series(out, dtype=float)


def ceiling_table(per_race: dict, variant="lodo") -> pd.DataFrame:
    """The noise floor: order and gap measures of the judge's whole-race replay at each
    CEILING_LEVELS level, per race and pooled (close pairs pooled, the rest means over races)."""
    rows = []
    for race, t in sorted(per_race.items()):
        lp, dr, o = t.get("a_laps"), t.get("a_drivers"), t.get("o_rows")
        if lp is None or dr is None or o is None or not len(lp) or not len(dr):
            continue
        lp, dr = lp[lp["variant"] == variant], dr[dr["variant"] == variant]
        fit = {"in": "in-sample", "lodo": "leave-one-driver-out"}[variant]
        base = o[(o["version"] == "1 judge") & (o["fit"] == fit) & (o["pace"] == "insample")]
        base = base.drop_duplicates("driver").set_index("driver")
        for level, parts in CEILING_LEVELS:
            err = ceiling_errors(lp, dr, parts).reindex(base.index)
            t2 = base.reset_index().assign(pred_total_s=base["real_s"].to_numpy() + err.to_numpy())
            m = order_measures(t2)
            if m:
                rows.append({"level": level, "race_id": race, **m})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    pooled = []
    for level, part in t.groupby("level", sort=False):
        pairs = part["close_pairs"].sum()
        pooled.append({"level": level, "race_id": "pooled",
                       "close_pair_acc": float((part["close_pair_acc"] * part["close_pairs"]).sum() / pairs) if pairs else np.nan,
                       "close_pairs": int(pairs),
                       **{c: _mean(part[c]) for c in ORDER_COLS if c not in ("close_pair_acc", "close_pairs")}})
    return pd.concat([t, pd.DataFrame(pooled)], ignore_index=True)[["level", "race_id"] + ORDER_COLS]


def ceiling_bootstrap(per_race: dict, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """Pooled close-pair order and relative gap error per ceiling level, with race-cluster
    bootstrap 95% intervals."""
    t = ceiling_table(per_race)
    if not len(t):
        return pd.DataFrame()
    t = t[t["race_id"] != "pooled"]
    races = sorted(t["race_id"].unique())
    rng = np.random.default_rng(seed)
    draws = [rng.choice(len(races), len(races)) for _ in range(n_boot)]
    out = []
    for level, part in t.groupby("level", sort=False):
        p = part.set_index("race_id").reindex(races)
        num = (p["close_pair_acc"].fillna(0) * p["close_pairs"]).to_numpy(dtype=float)
        den = p["close_pairs"].to_numpy(dtype=float)
        gap = p["rel_gap_mae_s"].to_numpy(dtype=float)
        acc_b = [num[i].sum() / den[i].sum() for i in draws if den[i].sum()]
        gap_b = [np.nanmean(gap[i]) for i in draws]
        out.append({"level": level, "close_pair_acc": num.sum() / den.sum(),
                    "close_pair_lo": float(np.percentile(acc_b, 2.5)), "close_pair_hi": float(np.percentile(acc_b, 97.5)),
                    "spearman": _mean(p["spearman"]), "rel_gap_mae_s": float(np.nanmean(gap)),
                    "rel_gap_lo": float(np.percentile(gap_b, 2.5)), "rel_gap_hi": float(np.percentile(gap_b, 97.5)),
                    "abs_mae_s": _mean(p["abs_mae_s"]), "close_pairs": int(den.sum()), "races": len(races)})
    return pd.DataFrame(out)


def _err_stats(x) -> dict:
    return {"n": int(np.isfinite(np.asarray(x, dtype=float)).sum()), "mae_s": _mae(x), "bias_s": _mean(x)}


def a_summary(per_race: dict) -> pd.DataFrame:
    """A1/A3 measures by variant, per race and pooled."""
    d, s, sg, lp = (concat(per_race, k) for k in ("a_drivers", "a_stints", "a_stop_gain", "a_laps"))
    rows = []
    for race in [*sorted(per_race), "pooled"]:
        def sel(t, race=race):
            return t if race == "pooled" or not len(t) else t[t["race_id"] == race]
        for v in ("in", "lodo", "loso"):
            dv = sel(d)[sel(d)["variant"] == v] if len(d) else d
            sv = sel(s)[sel(s)["variant"] == v] if len(s) else s
            gv = sel(sg)[sel(sg)["variant"] == v] if len(sg) else sg
            lv = sel(lp)[sel(lp)["variant"] == v] if len(lp) else lp
            if not (len(dv) or len(sv)):
                continue
            row = {"race_id": race, "variant": v}
            if len(dv):
                rel = dv["err_pace_s"] - dv.groupby("race_id")["err_pace_s"].transform("mean")
                row.update(race_n=len(dv), race_raw_mae_s=_mae(dv["err_s"]), race_raw_bias_s=_mean(dv["err_s"]),
                           race_pace_mae_s=_mae(dv["err_pace_s"]), race_pace_bias_s=_mean(dv["err_pace_s"]),
                           race_rel_mae_s=_mae(rel),
                           pit_err_per_stop_s=float(dv["pit_err_s"].sum() / max(dv["stops"].sum(), 1)),
                           pit_mae_per_stop_s=_mae(dv["pit_err_per_stop_s"]),
                           neutral_err_per_lap_s=float(dv["neutral_err_s"].sum() / dv["neutral_laps"].sum())
                           if dv["neutral_laps"].sum() else np.nan)
            if len(sv):
                row.update(stint_n=len(sv), stint_mae_s=_mae(sv["err_s"]), stint_bias_s=_mean(sv["err_s"]),
                           stint_mae_per_lap_s=_mean(np.abs(sv["err_s"]) / sv["laps"].clip(lower=1)))
            if len(gv):
                row.update(stop_gain_n=len(gv), stop_gain_mae_s=_mae(gv["err_s"]), stop_gain_bias_s=_mean(gv["err_s"]))
            if len(lv):
                late = lv[lv["clean"] & (lv["laps_to_stint_end"] < LATE_STINT_LAPS) & ~lv["kind"].isin(["in_lap"])]
                row.update(late_stint_resid_s=_mean(late["resid_s"]))
            rows.append(row)
    return pd.DataFrame(rows)


def lap_residuals(per_race: dict) -> pd.DataFrame:
    lp = concat(per_race, "a_laps")
    if not len(lp):
        return pd.DataFrame()
    c = lp[lp["clean"]].copy()
    c["age_bucket"] = c["tyre_age"].map(_bucket)
    g = c.groupby(["variant", "compound", "age_bucket"])["resid_s"]
    out = g.agg(n="count", mean_resid_s="mean").reset_index()
    out["order"] = out["age_bucket"].map(lambda b: int(b.split("-")[0].rstrip("+")))
    return out.sort_values(["variant", "compound", "order"]).drop(columns="order")


def neutral_residuals(per_race: dict) -> pd.DataFrame:
    lp = concat(per_race, "a_laps")
    if not len(lp):
        return pd.DataFrame()
    n = lp[lp["kind"].isin(["sc", "vsc"])]
    return n.groupby(["variant", "kind"])["resid_s"].agg(n="count", mean_resid_s="mean",
                                                       mae_s=lambda x: _mae(x)).reset_index()


def a4_summary(per_race: dict) -> pd.DataFrame:
    rows = []
    refs, brefs, meta = concat(per_race, "a_refs"), concat(per_race, "b_refs"), concat(per_race, "a_meta")
    for race in sorted(per_race):
        r = refs[refs["race_id"] == race] if len(refs) else refs
        b = brefs[brefs["race_id"] == race] if len(brefs) else brefs
        m = meta[meta["race_id"] == race]
        mode = m["field_mode"].iloc[0] if len(m) else None
        rs = r["ref_stops"].dropna()
        rows.append({"race_id": race, "field_mode": mode, "ref_mode": cross_race.stop_mode(rs),
                     "ref_stop_counts": cross_race._counts(rs) if len(rs) else "",
                     "match_field_mode": _share(rs == mode) if mode is not None else np.nan,
                     "match_own_stops": _share(r["ref_stops"] == r["field_stops"]) if len(r) else np.nan,
                     "lap0_beyond_raced": _share(r.get("beyond_raced", pd.Series(dtype=float))),
                     "lap0_beyond_raced_margin": _share(r.get("beyond_raced_margin", pd.Series(dtype=float))),
                     "grid_refs": len(b),
                     "grid_beyond_raced": _share(b["beyond_raced"]) if "beyond_raced" in b else np.nan,
                     "grid_beyond_raced_margin": _share(b["beyond_raced_margin"]) if "beyond_raced_margin" in b else np.nan,
                     "panel_gap_to_ref_s": _mean(b["panel_best_w_s"] - b["ref_mean_s"]) if "ref_mean_s" in b else np.nan})
    return pd.DataFrame(rows)


def b_summary(per_race: dict) -> pd.DataFrame:
    rows_all, snaps_all, fac_all, g_all = (concat(per_race, k) for k in ("b_rows", "b_snaps", "b_factors", "b_g"))
    out = []
    for race in [*sorted(per_race), "pooled"]:
        def sel(t):
            return t if race == "pooled" or not len(t) else t[t["race_id"] == race]
        r, s, f, g = sel(rows_all), sel(snaps_all), sel(fac_all), sel(g_all)
        rv = r[r["valid"]] if len(r) else r
        row = {"race_id": race, "snapshots": len(s),
               "rank_rho": _mean(s.get("rho", [])), "regret_s": _mean(s.get("regret_s", [])),
               "same_favourite": _share(s.get("same_favourite", pd.Series(dtype=float)).astype(float)),
               "fav_b_illegal_w": _share(s.get("fav_b_illegal_w", pd.Series(dtype=float)).astype(float)),
               "pit_now_agree": _share(s["pit_now_agree"].dropna().astype(float)) if "pit_now_agree" in s else np.nan,
               "rank_rho_green": _mean(s.get("rho_green", [])), "regret_green_s": _mean(s.get("regret_green_s", [])),
               "plans": len(rv), "b1_mae_s": _mae(rv["err_s"]) if len(rv) else np.nan,
               "b1_green_mae_s": _mae(rv["err_green_s"]) if len(rv) else np.nan,
               "b1_green_bias_s": _mean(rv["err_green_s"]) if len(rv) else np.nan,
               "b1_bias_s": _mean(rv["err_s"]) if len(rv) else np.nan,
               "b1_mae_per_lap_s": _mean(np.abs(rv["err_s"]) / rv["horizon"]) if len(rv) else np.nan}
        for c in ("cov_68", "cov_90", "gap_cov_68", "gap_cov_90"):
            row[c] = _share(rv[c].dropna().astype(float)) if c in rv else np.nan
        if len(f):
            row["factor_cov_68"] = _share(f["covered_68"].dropna().astype(float))
            row["factor_cov_90"] = _share(f["covered_90"].dropna().astype(float))
        if len(g):
            row.update(b4_mae_s=_mae(g["err_s"]), b4_bias_s=_mean(g["err_s"]), b4_mae_offset_s=_mae(g["err_offset_s"]),
                       b4_bias_offset_s=_mean(g["err_offset_s"]))
        out.append(row)
    return pd.DataFrame(out)


def b1_by(per_race: dict, by: str) -> pd.DataFrame:
    r = concat(per_race, "b_rows")
    if not len(r):
        return pd.DataFrame()
    r = r[r["valid"]].copy()
    if by == "horizon":
        r["horizon_bucket"] = (r["horizon"] // 10 * 10).astype(int)
        by = "horizon_bucket"
    return r.groupby(by)["err_s"].agg(n="count", mae_s=lambda x: _mae(x), bias_s="mean").reset_index()


def _pooled_source(src: str) -> str:
    """A component source for pooling across races: the count of stops seen dropped, and the
    snapshot at half distance (a different lap per race) named k(mid)."""
    import re
    src = re.sub(r", n=\d+", "", src)
    return re.sub(r"(@| to )k(\d+)", lambda m: m.group(0) if int(m.group(2)) in (0, 10, 20) else f"{m.group(1)}k(mid)", src)


def d_summary(per_race: dict) -> pd.DataFrame:
    d = concat(per_race, "d_rows")
    if not len(d):
        return pd.DataFrame()
    d = d.assign(source=d["source"].map(_pooled_source))
    rows = []
    for (comp, src, kind), part in d.groupby(["component", "source", "target_kind"], sort=False):
        rows.append({"component": comp, "source": src, "target_kind": kind, "n": int(part["err"].notna().sum()),
                     "races": int(part.loc[part["err"].notna(), "race_id"].nunique()), "mae": _mae(part["err"]),
                     "bias": _mean(part["err"]), "cov_68": _share(part["covered_68"].dropna().astype(float)),
                     "cov_90": _share(part["covered_90"].dropna().astype(float))})
    return pd.DataFrame(rows)


def h_summary(per_race: dict) -> pd.DataFrame:
    """Section A paired: each substitute against the judge as configured (in-sample and LODO)."""
    base = a_summary({r: {k: t[k] for k in ("a_drivers", "a_stints", "a_stop_gain", "a_laps") if k in t}
                      for r, t in per_race.items()})
    base_o = order_table(concat(per_race, "o_rows"))
    rows = []
    labels = []
    for t in per_race.values():
        for label, names, _ in t.get("h_variants", []):
            if label not in labels:
                labels.append(label)
    for label in labels:
        sub = {r: next((v for lab, _, v in t.get("h_variants", []) if lab == label), {}) for r, t in per_race.items()}
        errs = [r for r, v in sub.items() if "error" in v]
        a = a_summary({r: v for r, v in sub.items() if "a_drivers" in v})
        for variant in ("in", "lodo"):
            pa = a[(a["race_id"] == "pooled") & (a["variant"] == variant)]
            pb = base[(base["race_id"] == "pooled") & (base["variant"] == variant)]
            if not len(pa) or not len(pb):
                continue
            fit = {"in": "in-sample", "lodo": "leave-one-driver-out"}[variant]
            o_sub = base_o[(base_o["race_id"] == "pooled") & (base_o["label"] == label) & (base_o["fit"] == fit)
                           & (base_o["pace"] == "insample")]
            o_base = base_o[(base_o["race_id"] == "pooled") & (base_o["label"] == "as configured") & (base_o["fit"] == fit)
                            & (base_o["pace"] == "insample")]
            row = {"substitute": label, "variant": variant,
                   "replaces": SUBSTITUTES[label]["replaces"] if label in SUBSTITUTES else "every whole-race input above",
                   "with": SUBSTITUTES[label]["with"] if label in SUBSTITUTES else "every substitute together",
                   "races_failed": " ".join(errs)}
            row["close_pairs"] = int(o_sub["close_pairs"].iloc[0]) if len(o_sub) else 0
            row["close_pairs_base"] = int(o_base["close_pairs"].iloc[0]) if len(o_base) else 0
            for col in ("close_pair_acc", "spearman", "rel_gap_mae_s"):
                row[f"d_{col}"] = (float(o_sub[col].iloc[0] - o_base[col].iloc[0]) if len(o_sub) and len(o_base) else np.nan)
            for col in ("race_pace_mae_s", "race_pace_bias_s", "stop_gain_bias_s", "pit_err_per_stop_s", "stint_mae_s"):
                row[f"d_{col}"] = float(pa[col].iloc[0] - pb[col].iloc[0]) if col in pa and col in pb else np.nan
            if variant == "in":   # A4 has no fit variant: once per substitute
                a4b = a4_summary({r: {k: t[k] for k in ("a_refs", "a_meta") if k in t} for r, t in per_race.items()})
                a4v = a4_summary({r: {k: v[k] for k in ("a_refs", "a_meta") if k in v} for r, v in sub.items() if "a_refs" in v})
                for col in ("match_field_mode", "lap0_beyond_raced", "lap0_beyond_raced_margin"):
                    row[f"d_{col}"] = _mean(a4v[col]) - _mean(a4b[col]) if len(a4v) and len(a4b) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# --- headlines (also the ablation's measures) ----------------------------------------------------
HEADLINES = [   # (name, direction: +1 higher is better, -1 lower is better, 0 closer to 0 is better), primary first
    ("A_close_pair_acc", +1), ("A_spearman", +1), ("A_rel_gap_mae_s", -1),
    ("B2_rank_rho", +1), ("B2_regret_s", -1), ("B2_pit_now_agree", +1),
    ("P_close_pair_acc", +1), ("P_spearman", +1),
    ("A_abs_mae_s", -1), ("A_abs_bias_s", 0), ("A_stop_gain_bias_s", 0),
    ("B1_mae_s", -1), ("B1_bias_s", 0), ("B1_green_mae_s", -1), ("B2_rank_rho_green", +1),
    ("B3_cov_90", 0), ("B3_factor_cov_90", 0), ("B4_mae_offset_s", -1),
]
HEADLINE_DEF = {
    "A_close_pair_acc": "judge, leave-one-driver-out fit, whole-race pace offset: share of close pairs (same laps, real gap < 10 s) in the right order",
    "A_spearman": "same, Spearman of simulated against real finishing order",
    "A_rel_gap_mae_s": "same, mean |time error - the race's mean error| over cars on the lead lap (relative gap error)",
    "B2_rank_rho": "Spearman of predictive against judge remaining time over the jointly legal panel, mean over snapshots",
    "B2_regret_s": "judge time of the predictive favourite less the judge favourite, mean over snapshots",
    "B2_pit_now_agree": "share of snapshots where predictive and judge agree on pit-now against stay",
    "P_close_pair_acc": "predictive replay of every car's real remaining strategy (as-of fit and pace), close-pair order, mean over k",
    "P_spearman": "same, Spearman, mean over k",
    "A_abs_mae_s": "judge LODO pace-adjusted whole-race time, mean |error|",
    "A_abs_bias_s": "same, mean error",
    "A_stop_gain_bias_s": "judge LODO predicted less real gain at real stops",
    "B1_mae_s": "predictive less judge remaining time over the panel, mean |error|",
    "B1_bias_s": "same, mean",
    "B1_green_mae_s": "B1 with no SC/VSC on either side (the physics difference without the belief's event model)",
    "B2_rank_rho_green": "B2 rank agreement with no SC/VSC on either side",
    "B3_cov_90": "share of panel plans whose judge remaining time is inside the predictive 90% interval (target 0.90)",
    "B3_factor_cov_90": "share of fitted factors whose judge value is inside the predictive 90% interval (target 0.90)",
    "B4_mae_offset_s": "predictive replay of the real remaining strategy against reality, pace-offset error (G)",
}


def _ns(values) -> tuple[float, int]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.sum()), int(len(v))


def race_stats(tables: dict) -> dict:
    """{headline: (numerator, denominator)} of ONE race; headline() pools them as sum / sum, so a
    race-cluster bootstrap only resamples these. Means over races (Spearman, gap and time errors)
    enter as (value, 1)."""
    out = {}
    o = tables.get("o_rows")
    if o is not None and len(o):
        a = o[(o["version"] == "1 judge") & (o["fit"] == "leave-one-driver-out") & (o["pace"] == "insample")]
        m = order_measures(a) if len(a) else {}
        if m:
            out["A_close_pair_acc"] = (m["close_pair_acc"] * m["close_pairs"] if m["close_pairs"] else 0.0, m["close_pairs"])
            for k, col in (("A_spearman", "spearman"), ("A_rel_gap_mae_s", "rel_gap_mae_s"), ("A_abs_mae_s", "abs_mae_s"),
                           ("A_abs_bias_s", "bias_s")):
                out[k] = _ns([m[col]])
        p = o[(o["version"] == "3 predictive") & (o["pace"] == "asof")]
        ms = [order_measures(part) for _, part in p.groupby("k")]
        ms = [x for x in ms if x]
        if ms:
            out["P_close_pair_acc"] = (sum(x["close_pair_acc"] * x["close_pairs"] for x in ms if x["close_pairs"]),
                                       sum(x["close_pairs"] for x in ms))
            out["P_spearman"] = _ns([x["spearman"] for x in ms])
    sg = tables.get("a_stop_gain")
    if sg is not None and len(sg):
        out["A_stop_gain_bias_s"] = _ns(sg.loc[sg["variant"] == "lodo", "err_s"])
    s = tables.get("b_snaps")
    if s is not None and len(s):
        for k, col in (("B2_rank_rho", "rho"), ("B2_regret_s", "regret_s"), ("B2_pit_now_agree", "pit_now_agree"),
                       ("B2_rank_rho_green", "rho_green")):
            if col in s:
                out[k] = _ns(s[col].astype(float))
    r = tables.get("b_rows")
    if r is not None and len(r):
        r = r[r["valid"]]
        out["B1_mae_s"] = _ns(np.abs(r["err_s"]))
        out["B1_bias_s"] = _ns(r["err_s"])
        out["B1_green_mae_s"] = _ns(np.abs(r["err_green_s"]))
        if "cov_90" in r:
            out["B3_cov_90"] = _ns(r["cov_90"].astype(float))
    f = tables.get("b_factors")
    if f is not None and len(f):
        out["B3_factor_cov_90"] = _ns(f["covered_90"].astype(float))
    g = tables.get("b_g")
    if g is not None and len(g):
        out["B4_mae_offset_s"] = _ns(np.abs(g["err_offset_s"]))
    return out


def _pool(stats: list[dict]) -> dict:
    out = {}
    for k, _ in HEADLINES:
        num = sum(s[k][0] for s in stats if k in s)
        den = sum(s[k][1] for s in stats if k in s)
        out[k] = num / den if den else float("nan")
    return out


def headline(per_race: dict) -> dict:
    """The headline measures of one scorecard, pooled over its races."""
    return _pool([race_stats(t) for t in per_race.values()])


def three_distances(per_race: dict) -> pd.DataFrame:
    """Per race and pooled: judge-reality (in-sample and LODO), predictive-judge, predictive-reality."""
    o = order_table(concat(per_race, "o_rows"))
    b = b_summary(per_race)
    rows = []
    for race in [*sorted(per_race), "pooled"]:
        def oo(version, fit, pace, k=None):
            sel = o[(o["race_id"] == race) & (o["version"] == version) & (o["fit"] == fit) & (o["pace"] == pace)]
            return sel
        row = {"race_id": race}
        for tag, fit in (("JR_in", "in-sample"), ("JR_lodo", "leave-one-driver-out")):
            s = oo("1 judge", fit, "insample")
            if len(s):
                row[f"{tag}_close_pair_acc"] = float(s["close_pair_acc"].iloc[0])
                row[f"{tag}_spearman"] = float(s["spearman"].iloc[0])
                row[f"{tag}_rel_gap_mae_s"] = float(s["rel_gap_mae_s"].iloc[0])
                row[f"{tag}_abs_mae_s"] = float(s["abs_mae_s"].iloc[0])
        bb = b[b["race_id"] == race]
        if len(bb):
            row.update(PJ_rank_rho=float(bb["rank_rho"].iloc[0]), PJ_regret_s=float(bb["regret_s"].iloc[0]),
                       PJ_mae_s=float(bb["b1_mae_s"].iloc[0]), PJ_bias_s=float(bb["b1_bias_s"].iloc[0]))
        p = o[(o["race_id"] == race) & (o["version"] == "3 predictive") & (o["pace"] == "asof")]
        if len(p):
            row.update(PR_close_pair_acc=_mean(p["close_pair_acc"]), PR_spearman=_mean(p["spearman"]),
                       PR_rel_gap_mae_s=_mean(p["rel_gap_mae_s"]))
        if len(bb) and "b4_mae_offset_s" in bb:
            row["PR_g_mae_offset_s"] = float(bb["b4_mae_offset_s"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_delta(base: dict, arm: dict, n_boot=N_BOOT, seed=0) -> dict:
    """Per headline: arm less base, pooled, with a race-cluster bootstrap 95% interval (races
    resampled with replacement, both scorecards on the same draw)."""
    races = sorted(set(base) & set(arm))
    sb = {r: race_stats(base[r]) for r in races}
    sa = {r: race_stats(arm[r]) for r in races}
    hb, ha = _pool(list(sb.values())), _pool(list(sa.values()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        x, y = _pool([sb[r] for r in pick]), _pool([sa[r] for r in pick])
        draws.append({k: y[k] - x[k] for k, _ in HEADLINES})
    out = {}
    for k, _ in HEADLINES:
        v = np.array([d[k] for d in draws], dtype=float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
        out[k] = (ha[k] - hb[k], float(lo), float(hi))
    return out


# =================================================================================================
# writing
# =================================================================================================
def _md(df, digits=2) -> str:
    return cross_race.md_table(df.reset_index(drop=True) if df is not None else df, digits)


FRAMEWORK = """There are two simulators. The JUDGE (the eval's "world") scores every plan and builds the
reference: it should be as close as possible to what really happened while using as little
hindsight as it can. The PREDICTIVE simulator (the "belief") is what the agent, its tools and the
rule baselines plan with at a snapshot: it uses no hindsight, and should be as close as possible to
the judge under that restriction. Finishing ORDER leads every section: close-competitor pair order
first, then Spearman and position error, then relative gaps (common offset removed), then absolute
time. Selection rule for the ablation, stated before the results: primary = close-pair order
accuracy of the judge out of sample (A_close_pair_acc); then relative gap error (A_rel_gap_mae_s);
then plan-ranking agreement between predictive and judge (B2_rank_rho, B2_regret_s); absolute time
error is a tie-breaker. A factor that improves order or relative gaps while worsening absolute time
counts as an improvement."""


def write_scorecard(label: str, per_race: dict, spec: Spec, opts: Options, out_dir: Path, b5=None, secs=None,
                    failures=()) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    o = order_table(concat(per_race, "o_rows"))
    tables = {"order.csv": o, "three_distances.csv": three_distances(per_race), "a_summary.csv": a_summary(per_race),
              "a_drivers.csv": concat(per_race, "a_drivers"), "a_stints.csv": concat(per_race, "a_stints"),
              "a_stop_gain.csv": concat(per_race, "a_stop_gain"), "a_lap_residuals.csv": lap_residuals(per_race),
              "a_neutral.csv": neutral_residuals(per_race), "a4_decision.csv": a4_summary(per_race),
              "a_refs.csv": concat(per_race, "a_refs"), "a_hindsight_pairs.csv": h_summary(per_race),
              "b_summary.csv": b_summary(per_race), "b_snapshots.csv": concat(per_race, "b_snaps"),
              "b_rows.csv.gz": concat(per_race, "b_rows"), "b_factors.csv": concat(per_race, "b_factors"),
              "b_g.csv": concat(per_race, "b_g"), "d_components.csv": d_summary(per_race),
              "d_rows.csv": concat(per_race, "d_rows"), "d_status.csv": concat(per_race, "d_status"),
              "ledger.csv": ledger_frame(spec), "field_replay.csv": concat(per_race, "o_field_replay"),
              "ceiling.csv": ceiling_table(per_race), "ceiling_pooled.csv": ceiling_bootstrap(per_race)}
    for name, t in tables.items():
        if t is not None and len(t):
            # gzip without a timestamp, so an unchanged table writes an unchanged file
            t.to_csv(out_dir / name, index=False,
                     compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None)
    h = headline(per_race)
    meta = {"label": label, "races": sorted(per_race), "failed": list(failures), "code_version": CODE_VERSION,
            "world": "race default (jobs.load_race)" if spec.world is None else "file", "world_overlay": spec.world_overlay,
            "belief_overlay": spec.belief_overlay, "world_race": spec.world_race,
            "world_config_hash": config_key(spec.world_config()), "belief_config_hash": config_key(spec.belief_config()),
            "options": asdict(opts), "headline": h, "future_blindness": b5, "run_seconds": secs}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    md = scorecard_md(label, per_race, spec, opts, b5, secs)
    if failures:
        md += "\nFailed and left out: " + "; ".join(f"{x['race_id']} ({x['error']})" for x in failures) + "\n"
    (out_dir / "scorecard.md").write_text(md)
    return out_dir


def _order_view(o, race="pooled"):
    v = o[o["race_id"] == race].copy()
    return v[["version", "label", "k", "fit", "pace", "in_sample", *ORDER_COLS]]


def scorecard_md(label, per_race, spec, opts, b5=None, secs=None) -> str:
    o = order_table(concat(per_race, "o_rows"))
    led = ledger_frame(spec)
    a = a_summary(per_race)
    b = b_summary(per_race)
    lines = [f"# Simulator fidelity scorecard: {label}", "",
             f"Races: {', '.join(sorted(per_race))}. Code version {CODE_VERSION}. World config "
             f"{config_key(spec.world_config())} ({'race default' if spec.world is None else 'file'}"
             f"{', overlay ' + json.dumps(spec.world_overlay) if spec.world_overlay else ''}"
             f"{', RaceParams ' + json.dumps(spec.world_race) if spec.world_race else ''}); belief config "
             f"{config_key(spec.belief_config())}{', overlay ' + json.dumps(spec.belief_overlay) if spec.belief_overlay else ''}. "
             f"Options: {json.dumps(asdict(opts))}." + (f" Run time {secs:.0f} s." if secs else ""), "",
             FRAMEWORK, "",
             "## Headline", "", _md(pd.DataFrame([{"measure": k, "value": headline(per_race).get(k, np.nan),
                                                    "definition": HEADLINE_DEF[k]} for k, _ in HEADLINES]), 3),
             "## The three distances (per race and pooled)", "",
             "JR = judge against reality (version 1; in = in-sample fit, lodo = leave-one-driver-out fit; pace = the "
             "car's whole-race median clean-lap offset in both). PJ = predictive against judge on the plan panel (B). "
             "PR = predictive against reality (version 3, as-of fit and pace, mean over k; G = the existing midrace "
             "measure, pace-offset error).", "", _md(three_distances(per_race), 3),
             "## O. Order and time, three versions (pooled)", "",
             "fit: what the tyre parameters were fitted on (in-sample: every car's laps; leave-one-driver-out: every car "
             "but the one predicted; as-of: race laps up to k shrunk to practice, the predictive simulator). pace: the "
             "car's pace term (insample: its whole-race median clean-lap residual, one number per car, reads the car's "
             "own laps; practice: this weekend's practice long runs, before the race; asof: its green laps up to k, "
             "practice where fewer than five; none: the field-mean car). in_sample: either reads the car's laps after k. "
             "Versions 1 and 2 replay from lap 0; version 3 from k. Without traffic the order is the order of "
             "simulated times; it becomes a traffic measure when the traffic branch lands.", "",
             _md(_order_view(o), 3)]
    lines += ["## Noise floor (the judge handed the real race, step by step)", "",
              "The judge's whole-race replay (version 1, leave-one-driver-out fit, whole-race pace offset: level 0 is "
              "the headline) with each level's real values in place of the model's, cumulative from 1 to 4, then each "
              "alone: " + "; ".join(f"{k}: {v}" for k, v in CEILING_PARTS.items()) + ". What level 4 leaves is "
              "lap-to-lap noise around each stint's pace, incidents, errors and unmodelled effects. Pooled with "
              "race-cluster bootstrap 95% intervals:", "", _md(ceiling_bootstrap(per_race), 3)]
    if "o_field_replay" in next(iter(per_race.values()), {}):
        lines += ["Whole-field replay of branch feat/traffic-in-eval (its own summary):", "",
                  _md(concat(per_race, "o_field_replay"), 3)]
    lines += ["## A. Judge versus reality", "",
              "A1/A3 by fit variant (race_* per car over the whole race; pace = the car's median clean-lap offset; "
              "stint = every real stint less lap 1, in- and out-laps; pit = error on in+out laps with the charged "
              "loss, per stop; neutral = error per SC/VSC lap; late_stint = clean laps within 5 of the stint end). "
              "loso rows are the held-out stint only.", "", _md(a[a["race_id"] == "pooled"], 3),
              "Per race:", "", _md(a[a["race_id"] != "pooled"], 2),
              "Lap residuals (real less predicted, pace removed) on clean laps by compound and tyre age, pooled:", "",
              _md(lap_residuals(per_race).pivot_table(index=["variant", "compound"], columns="age_bucket",
                                                      values="mean_resid_s").reset_index()
                  if len(lap_residuals(per_race)) else None, 2),
              "Neutralised laps:", "", _md(neutral_residuals(per_race), 2),
              "A4: the judge's lap-0 reference against the field (and the share of reference plans, at lap 0 and at "
              "every B snapshot, with a stint beyond the oldest age any car ran on that compound, or beyond it plus "
              "the R8 margin):", "", _md(a4_summary(per_race), 2),
              "A paired: each whole-race input replaced by its no-hindsight substitute, change against the judge as "
              "configured (order: pace insample; positive d_close_pair_acc is better):", "", _md(h_summary(per_race), 3),
              "A6: hindsight ledger (judge and predictive):", "", _md(led[["role", "id", "input", "tag", "substitute", "source"]]),
              "## B. Predictive versus judge", "",
              "Plan panel per snapshot: the real remaining strategy, stay out, one stop at "
              f"{opts.n_one} laps spread from the soonest legal stop on each compound both model, two stops at "
              f"{opts.n_two} first-stop laps on every compound pair. Times under R1-R3 (both simulators); ranking (B2) "
              "over the plans legal under both configs' own rules. Snapshots every "
              f"{opts.b_every} laps for {opts.b_drivers} sampled drivers per race. {max(1, opts.samples) if opts.samples else COVERAGE_SAMPLES} "
              "draw(s) per plan; the belief draws from seed + BELIEF_SEED_OFFSET.", "",
              _md(b, 3), "B1 by horizon and by race phase:", "", _md(b1_by(per_race, "horizon"), 2),
              _md(b1_by(per_race, "phase"), 2), "B1 by plan kind:", "", _md(b1_by(per_race, "kind"), 2),
              f"B5 future blindness: {json.dumps(b5) if b5 else 'not run'}", "",
              "## D. Component targets (pooled over races)", "",
              "Each component's prediction from allowed information against the whole-race measurement of the same "
              "quantity. err = pred - target; coverage only where the predictor states an sd.", "",
              _md(d_summary(per_race), 3), "Component status:", "",
              _md(concat(per_race, "d_status").groupby(["component", "status"]).size().reset_index(name="races")
                  if len(concat(per_race, "d_status")) else None)]
    return "\n".join(lines) + "\n"


# =================================================================================================
# ablation
# =================================================================================================
def load_ablation(path=ABLATION_SPEC) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def _factor_overlays(f: dict) -> tuple[dict, dict]:
    """(world overlay, belief overlay) of an ablation factor: `both`, then `world` / `belief`, and
    `world_file` / `belief_file` (a config file read as an overlay)."""
    both = f.get("both") or {}
    w, b = merge(both, f.get("world") or {}), merge(both, f.get("belief") or {})
    if f.get("world_file"):
        w = merge(w, read_config_file(ROOT / f["world_file"])[0])
    if f.get("belief_file"):
        b = merge(b, read_config_file(ROOT / f["belief_file"])[0])
    return w, b


def factor_absent(f: dict) -> str | None:
    """Why a factor's switch is not on this branch (None when it is): a module in `requires` that
    does not import, a file it names that does not exist, or an overlay SimConfig rejects (an
    unknown field: the branch that adds it has not merged)."""
    for mod in f.get("requires") or ():
        if importlib.util.find_spec(mod) is None:
            return f"module {mod} not on this branch"
    for key in ("world_file", "belief_file"):
        if f.get(key) and not (ROOT / f[key]).exists():
            return f"{f[key]} not on this branch"
    try:
        w, b = _factor_overlays(f)
        SimConfig.model_validate(merge(SimConfig().model_dump(), w))
        SimConfig.model_validate(merge(SimConfig().model_dump(), b))
    except Exception as exc:
        return f"switch not on this branch ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})"
    return None


def ablation_specs(base: Spec, ab: dict, skipped=None, only=None) -> list[Spec]:
    """The base, each factor present on this branch alone, and all_on (its present factors). A
    factor whose switch is absent (factor_absent) is skipped and listed in `skipped`. `only`
    (factor names, "all_on" among them to keep it) runs those alone beside the base; an unknown
    name is an error."""
    specs = [replace(base, label="base")]
    factors = every = ab.get("factors") or {}
    if only:
        unknown = sorted(set(only) - set(factors) - {"all_on"})
        if unknown:
            raise ValueError(f"--only: unknown factors {unknown}; the spec has {sorted(factors)} and all_on")
        if "all_on" not in only:
            ab = {**ab, "all_on": []}
        factors = {n: f for n, f in factors.items() if n in only}
    absent = {n: why for n, f in every.items() if (why := factor_absent(f))}
    if skipped is not None:
        skipped.extend({"factor": n, "reason": why} for n, why in absent.items())

    def apply(spec, name):
        f = every[name]
        w, b = _factor_overlays(f)
        return spec.with_factor(spec.label, world=w, belief=b, world_race=f.get("world_race"))
    for name in factors:
        if name not in absent:
            specs.append(replace(apply(base, name), label=name))
    on = [n for n in ab.get("all_on") or () if n not in absent]
    if on:
        s = base
        for name in on:
            s = apply(s, name)
        specs.append(replace(s, label="all_on"))
    return specs


def ablation_table(results: dict, specs: list[Spec], n_boot=N_BOOT) -> pd.DataFrame:
    base = results["base"]
    led_base = ledger_frame(specs[0])
    key = set(zip(led_base["role"], led_base["id"], led_base["tag"]))
    rows = []
    hb = headline(base)
    rows.append({"factor": "base", **{k: f"{hb.get(k, np.nan):.3f}" for k, _ in HEADLINES}, "ledger_change": ""})
    for s in specs[1:]:
        d = bootstrap_delta(base, results[s.label], n_boot=n_boot)
        led = ledger_frame(s)
        k2 = set(zip(led["role"], led["id"], led["tag"]))
        change = [f"+{r[0][0]}:{r[1]}({r[2]})" for r in sorted(k2 - key)] + [f"-{r[0][0]}:{r[1]}" for r in sorted(key - k2)
                                                                            if (r[0], r[1]) not in {(x[0], x[1]) for x in k2}]
        rows.append({"factor": s.label, **{k: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in d.items()},
                     "ledger_change": " ".join(change)})
    return pd.DataFrame(rows)


def ablation_md(table: pd.DataFrame, ab: dict, races, secs=None, skipped=(), failures=()) -> str:
    lines = ["# Simulator fidelity ablation", "", FRAMEWORK, "",
             f"Races: {', '.join(races)}. Rows: the base, each factor switched alone, and all_on "
             f"({', '.join(ab.get('all_on') or [])}). Cells: change against the base, pooled, with a race-cluster "
             f"bootstrap 95% interval ({N_BOOT} draws over the {len(races)} races every configuration ran)."
             + (f" Run time {secs:.0f} s." if secs else ""), "",
             "Factors:", ""]
    for name, f in (ab.get("factors") or {}).items():
        lines.append(f"- `{name}`: {f.get('note', '')} {json.dumps({k: v for k, v in f.items() if k != 'note'})}")
    if skipped:
        lines += ["", "Skipped (switch not on this branch):", ""]
        lines += [f"- `{x['factor']}`: {x['reason']}" for x in skipped]
    if failures:
        lines += ["", "Failed and left out (configuration, race, error):", ""]
        lines += [f"- {x['label']} {x['race_id']}: {x['error']}" for x in failures]
    lines += ["", "Measures (primary first; direction: + better for acc/rho/agree, - better for errors):", ""]
    lines += [f"- `{k}`: {HEADLINE_DEF[k]}" for k, _ in HEADLINES]
    lines += ["", _md(table, 3)]
    return "\n".join(lines) + "\n"


# =================================================================================================
# worker processes
# =================================================================================================
# Every worker runs one BLAS thread: the races are the parallelism, and N workers each spinning up
# a full BLAS pool oversubscribe a shared machine. Workers are spawned (not forked) with these set
# in the environment, so each one imports numpy fresh under them; the initialiser sets them again.
from concurrent.futures import ProcessPoolExecutor as _ProcessPool   # noqa: E402

THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DEFAULT_JOBS = min(6, max(1, (os.cpu_count() or 2) // 2))


def _worker_init():
    os.environ.update(THREAD_ENV)


class _OneThreadPool(_ProcessPool):
    """The process pool run_specs uses (it reads the module name ProcessPoolExecutor, bound to this
    class below): spawned workers, one BLAS thread each, at most DEFAULT_JOBS unless asked."""

    def __init__(self, max_workers=None, **kw):
        import multiprocessing
        os.environ.update(THREAD_ENV)   # inherited by the spawned workers before numpy loads
        super().__init__(max_workers=max_workers or DEFAULT_JOBS, mp_context=multiprocessing.get_context("spawn"),
                         initializer=_worker_init, **kw)


# bound here, below the cache key's code (CODE_VERSION), so the pool can change without invalidating
# cached results: it changes how races are scheduled, never what a race computes
ProcessPoolExecutor = _OneThreadPool


# =================================================================================================
# CLI
# =================================================================================================
def _options(args) -> Options:
    return Options(b_drivers=args.drivers, b_every=args.every, samples=0 if args.coverage else args.samples,
                   n_one=args.n_one, n_two=args.n_two, lodo=not args.no_lodo, loso=not args.no_loso,
                   grid_references=not args.no_grid_references, hindsight_variants=not args.no_hindsight_variants,
                   components=not args.no_components, seed=args.seed, field_replay=args.field_replay)


def _common(ap):
    ap.add_argument("--races", nargs="+", default=CALIBRATION_RACES)
    ap.add_argument("--world", default=DEFAULT_WORLD, help="SimConfig file, or 'default' (the race's own config)")
    ap.add_argument("--belief", default=str(BELIEF_FILE))
    ap.add_argument("--world-overlay", nargs="*", default=[])
    ap.add_argument("--belief-overlay", nargs="*", default=[])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--cache", default=None, help="cache directory, default <out>/.cache; 'none' disables")
    ap.add_argument("--drivers", type=int, default=6, help="B: drivers sampled per race")
    ap.add_argument("--every", type=int, default=5, help="B: snapshot every N laps")
    ap.add_argument("--samples", type=int, default=20, help="draws per simulation in B and version 3 (Options.samples)")
    ap.add_argument("--coverage", action="store_true", help=f"B3 interval coverage: {COVERAGE_SAMPLES} draws")
    ap.add_argument("--n-one", type=int, default=4)
    ap.add_argument("--n-two", type=int, default=3)
    ap.add_argument("--no-lodo", action="store_true")
    ap.add_argument("--no-loso", action="store_true")
    ap.add_argument("--no-grid-references", action="store_true")
    ap.add_argument("--no-hindsight-variants", action="store_true")
    ap.add_argument("--no-components", action="store_true")
    ap.add_argument("--no-future-blindness", action="store_true")
    ap.add_argument("--field-replay", action="store_true",
                    help="also run src/sim/field_replay.py's whole-field replay and report its summary")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"worker processes, one BLAS thread each (default min(6, cpu_count // 2) = {DEFAULT_JOBS})")


def _cache_dir(args):
    if args.cache == "none":
        return None
    return Path(args.cache) if args.cache else Path(args.out) / ".cache"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.fidelity", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the scorecard for one configuration")
    _common(r)
    r.add_argument("--label", default="main")
    rp = sub.add_parser("report", help="print a written scorecard or ablation")
    rp.add_argument("--out", default=str(OUT_DIR))
    rp.add_argument("--label", default="main")
    a = sub.add_parser("ablate", help="base, each factor alone, all on: one table")
    _common(a)
    a.add_argument("--spec", default=str(ABLATION_SPEC))
    a.add_argument("--only", nargs="+", default=None,
                   help="run only these factors of the spec beside the base (add all_on to keep it)")
    a.add_argument("--n-boot", type=int, default=N_BOOT)
    lg = sub.add_parser("ledger", help="print the hindsight ledger of a configuration")
    lg.add_argument("--world", default=DEFAULT_WORLD)
    lg.add_argument("--belief", default=str(BELIEF_FILE))
    lg.add_argument("--world-overlay", nargs="*", default=[])
    lg.add_argument("--belief-overlay", nargs="*", default=[])
    args = ap.parse_args(argv)

    if args.cmd == "ledger":
        spec = make_spec(args.world, args.belief, args.world_overlay, args.belief_overlay)
        print(_md(ledger_frame(spec)))
        return 0
    if args.cmd == "report":
        for p in (Path(args.out) / args.label / "scorecard.md", Path(args.out) / "ablation" / "ablation.md"):
            if p.exists():
                print(p.read_text())
        return 0
    races = guard(args.races)
    spec = make_spec(args.world, args.belief, args.world_overlay, args.belief_overlay,
                     label=getattr(args, "label", "main"))
    opts = _options(args)
    cache = _cache_dir(args)
    t0 = time.time()
    if args.cmd == "run":
        print(f"fidelity run {spec.label}: {len(races)} races")
        failures = []
        res = run_specs([spec], races, opts, cache, args.jobs, failures=failures)
        b5 = None if args.no_future_blindness else future_blindness(spec)
        path = write_scorecard(spec.label, res[spec.label], spec, opts, Path(args.out) / spec.label, b5, time.time() - t0,
                               failures=failures)
        print((path / "scorecard.md").read_text())
        print(f"wrote {path} in {time.time() - t0:.0f} s")
        return 0
    ab = load_ablation(args.spec)
    skipped, failures = [], []
    specs = ablation_specs(replace(spec, label="base"), ab, skipped, only=args.only)
    for sk in skipped:
        print(f"  factor {sk['factor']} skipped: {sk['reason']}")
    print(f"fidelity ablate: {len(specs)} configurations x {len(races)} races")
    res = run_specs(specs, races, opts, cache, args.jobs, failures=failures)
    out = Path(args.out) / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    table = ablation_table(res, specs, n_boot=args.n_boot)
    table.to_csv(out / "ablation.csv", index=False)
    rows = []
    for s in specs:
        rows.append({"factor": s.label, **headline(res[s.label])})
    pd.DataFrame(rows).to_csv(out / "headlines.csv", index=False)
    secs = time.time() - t0
    used = sorted(set.intersection(*(set(res[s.label]) for s in specs)) if specs else set())
    (out / "ablation.md").write_text(ablation_md(table, ab, used, secs, skipped=skipped, failures=failures))
    print((out / "ablation.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
