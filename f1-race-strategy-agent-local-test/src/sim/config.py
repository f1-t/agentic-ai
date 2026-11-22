"""Simulator configuration. Components are chosen by registry key plus params."""
import json
import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


log = logging.getLogger(__name__)

# tyre age past which wear may steepen: the filter's reference model (data.pace_design) and
# the linear_fuel_hinge degradation model bend here unless SimConfig.hinge_knots says otherwise
HINGE_KNOT = 25
HINGE_TAU_S_PER_LAP = 0.05   # to_snapshot prior sd of a hinge slope, SimConfig.shrinkage["hinge_s_per_lap"]
# a lap is in free air when the car ahead on the same lap count crossed the line at least this
# long before us at the lap's start (data.lap_frame free_air). Set a priori, between the DRS
# detection gap (1 s) and the dirty-air window; Spain only, selected across races by LORO later.
FREE_AIR_GAP_S = 1.5

DEFAULT_COMPOUND_CLASSES = {
    **dict.fromkeys(["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"], "SLICK"),
    "INTERMEDIATE": "INTERMEDIATE", "WET": "WET",
}


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Cfg):
    name: str
    params: dict = Field(default_factory=dict)


class EventEffects(_Cfg):
    """How neutralised laps change lap time and pit loss. Provisional defaults."""

    sc_lap_factor: float = 1.36       # provisional, 2021 Spain observed mean (1.361, v2 filter); fitted per race in sprint 2
    vsc_lap_factor: float = 1.3       # VSC lap time as a multiple of the green lap
    sc_pit_loss_factor: float = 0.5   # share of the normal pit loss paid under SC
    vsc_pit_loss_factor: float = 0.6
    lap1_extra_s: float = 9.0         # provisional: standing start extra on lap 1, 2021 Spain median +8.7 to +10 s
    # tyre age gained per neutralised lap (1 per green lap), so a set run under SC/VSC is
    # younger afterwards. Provisional, fitted in sprint 2 from annotated races.
    sc_age_increment: float = 0.3
    vsc_age_increment: float = 0.5


class EventModel(_Cfg):
    """Rates for the "rate" event model (events.py), its ComponentSpec params.

    p_* is the chance per green lap that a period starts, d_* the mean period length in
    laps. A race whose RaceParams carries circuit rates (sc_rate_per_lap and friends)
    overrides these per race. Provisional, Spain only: one SC of 3 laps in 66 gives
    p_sc 0.015 and d_sc 3; the VSC values are placeholders. Fitted per circuit and era by
    event_rates (SPEC 0011b), which fills the RaceParams rates; these stay the no-data prior.

    p_sc_lap1 / p_vsc_lap1 (SPEC 0011b) replace p_sc / p_vsc on lap 1 alone, and only when
    the walk starts there (a snapshot at laps_completed 0). None, the default, leaves lap 1
    on p like every lap. event_rates estimates them as the share of races with a period of
    that kind starting on lap 1.
    """

    p_sc: float = Field(default=0.015, ge=0, le=1)
    d_sc: float = Field(default=3.0, ge=1)
    p_vsc: float = Field(default=0.02, ge=0, le=1)
    d_vsc: float = Field(default=2.0, ge=1)
    p_sc_lap1: float | None = Field(default=None, ge=0, le=1)
    p_vsc_lap1: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self):
        if self.p_sc + self.p_vsc > 1:
            raise ValueError("p_sc + p_vsc must be at most 1")
        lap1 = (self.p_sc if self.p_sc_lap1 is None else self.p_sc_lap1) + \
            (self.p_vsc if self.p_vsc_lap1 is None else self.p_vsc_lap1)
        if lap1 > 1:
            raise ValueError("the lap-1 rates p_sc_lap1 + p_vsc_lap1 must be at most 1")
        return self


class HazardEffects(_Cfg):
    """What a tyre hazard event costs (hazard.py gives the per lap probability). Provisional."""

    dnf_share: float = Field(default=0.5, ge=0, le=1)  # share of events that end the race
    # added once to a DNF sample, on top of its remaining laps run as if it had continued.
    # An arbitrary time-metric convention that only keeps a DNF slower than finishing; a
    # position metric uses BatchResult.dnf_lap (the lap of retirement) instead.
    dnf_penalty_s: float = 300.0
    big_loss_s: float = 20.0          # added on the event lap when the car carries on


class LongRun(_Cfg):
    """When a practice stint is a long run (data.practice_frame). Timed laps are green,
    non pit laps with a lap time, tyre age and compound, so the out and in laps never count."""

    min_timed_laps: int = 6           # qualifying runs are short
    # consistent: at least consistent_share of the timed laps within consistent_band_s of the
    # stint median. A push/cool programme has about half its timed laps off-pace; a long run
    # with a traffic lap or two has at most one in five; 0.8 encodes "at most one lap in five
    # off-pace". Frozen on a priori grounds, Spain only; selected across races by
    # leave-one-race-out later.
    consistent_share: float = 0.8
    consistent_band_s: float = 2.0
    max_pace_gap_s: float = 2.5       # stint median within this of the session's long-run reference


COMPOUND_STEPS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "compound_steps.json")


def load_compound_steps(path=COMPOUND_STEPS_FILE) -> dict:
    """The tier-5 neighbour steps (proxy.py) from the committed table, {"SOFT-MEDIUM": {...},
    "MEDIUM-HARD": {...}}: per pair pace_frac, wear_ratio and life_ratio, each {value, sd, n}.
    Regenerated by `python -m src.sim.compound_steps estimate` (configs/sim/compound_steps.json
    records how). Empty when the file is absent, so tier 5 is then unavailable."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        table = json.load(f)
    def num(x):   # the table writes a missing sd (one race) as null
        return float("nan") if x is None else float(x)
    return {pair: {k: {"value": num(v["value"]), "sd": num(v["sd"]), "n": int(v["n"])}
                   for k, v in entry.items() if k in ("pace_frac", "wear_ratio", "life_ratio")}
            for pair, entry in table.get("steps", {}).items()}


def _table_rows(key, path=COMPOUND_STEPS_FILE) -> list:
    """The rows of list section `key` of the committed table ("per_race": each calibration
    race's step row; "validation_rows": the forward-chained hidden-compound validation), the
    numbers proxy.as_of_config recomputes the tier-5 inputs from; [] when absent."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f).get(key) or []
    keep = ("race_id", "pair", "thin", *("pace_frac", "wear_ratio", "life_ratio"), "compound", "case", "available",
            "pace_err_s", "wear_err_s_per_lap")
    return [{k: r[k] for k in keep if k in r} for r in rows]


def _blend_section(key, path=COMPOUND_STEPS_FILE) -> dict:
    """{"pace": {...}, "wear": {...}} section `key` of the committed table ("weights": per
    source 1 / its validation mean squared error; "sd_floor": the blend's leave-one-race-out
    error), {} when absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        table = json.load(f)
    sec = table.get(key) or {}
    if key == "weights":
        return {k: {s: float(w) for s, w in v.items()} for k, v in sec.items()}
    return {k: float(v) for k, v in sec.items()}


class PolicyView(_Cfg):
    """What a windowed fit (to_snapshot, and the calibrator's laps_so_far) may read of the race
    frame at a snapshot (driver D after lap L; docs/eval.md "Information contract"). The full
    window (the world) reads every row whatever this says.

    window_cut   "time": the rows completed at or before the session time D completed lap L
                 (end_s <= t); another car's lap L that ended later is not visible, a car a lap
                 ahead's lap L + 1 that ended earlier is. "lap": every row with lap <= L, as
                 before (legacy).
    clean        "as_of": clean, pace_resid_s, driver_offset_s and the gap columns recomputed
                 on the visible rows alone (data.as_of_columns). "race": the lap frame's own,
                 whose robust filter saw every lap of the race (legacy).
    The defaults are leak-free; {"window_cut": "lap", "clean": "race"} reproduces the belief as
    it was before."""
    window_cut: Literal["time", "lap"] = "time"
    clean: Literal["as_of", "race"] = "as_of"


class CompoundProxy(_Cfg):
    """Pricing of dry compounds the race laps do not model (proxy.py, docs/eval.md "Proxied
    compounds"). With `enabled` False the simulator is exactly what it was before proxying.

    A compound with race laps (tier 1) keeps its fit. For the others pace and wear are each a
    precision-weighted blend, from the nearest raced compound, of every source in `sources` that
    has an estimate: practice long runs, qualifying (pace only) and the cross-race neighbour
    step (`steps`). `weights` ({"pace"|"wear": {source: 1 / mean squared error}}) are fixed, one
    set for all races; `sd_floor` ({"pace"|"wear": the blend's leave-one-race-out error}) is the
    smallest prior sd of a blended value (the cross-race spread raises it). All three are read
    from configs/sim/compound_steps.json, so config_hash covers them. `min_wear_s_per_lap`
    floors a proxied wear (a tyre does not get faster with age)."""

    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["practice", "qualifying", "cross_race_step"])
    min_wear_s_per_lap: float = 0.0
    steps: dict = Field(default_factory=load_compound_steps)
    weights: dict = Field(default_factory=lambda: _blend_section("weights"))
    sd_floor: dict = Field(default_factory=lambda: _blend_section("sd_floor"))
    # a windowed fit (a policy's) pricing race R reads steps, weights and sd_floor recomputed from
    # the races before R in `per_race` and `validation` (proxy.as_of_config); False reads the
    # whole-set table as before (legacy). The full window (the world) always reads the whole set.
    as_of: bool = True
    per_race: list = Field(default_factory=lambda: _table_rows("per_race"))
    validation: list = Field(default_factory=lambda: _table_rows("validation_rows"))


PIT_LOSS_PRIOR_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "pit_loss_prior.json")


def _load_pit_loss_prior(path=PIT_LOSS_PRIOR_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class FuelAnchor(_Cfg):
    """Fuel anchoring (docs/eval.md "Fuel anchoring"): within a stint tyre age and race lap rise
    together, so the fuel slope and the wear slopes are confounded. The anchored prior
    (fuel_prior.anchored_prediction: the lap-scaled physics slope times the as-of ratio of earlier
    races) pins the fuel term.

    prior     the as_of_history calibrator's fuel prior (race_history.as_of_fuel mode):
              "circuit_history" (as before, #102) or "anchored"; a calibrator param `fuel`
              overrides it.

    practice  "config" (as before): the practice fit's in-stint fuel, the column fixed at
              pre_race_fuel_slope_s_per_lap; "anchored": at the anchored prior of the race
              (race_history.as_of_fuel mode "anchored"), so the practice wear the to_snapshot fit
              shrinks toward is net of the same fuel the race fit is shrunk toward. Policy windows.
    full_fit  the full window (the judge): the race fuel slope shrunk toward the a priori physics
              prior (fuel_prior.physics_prediction, the race's laps and practice lap) with its sd,
              by a penalty residual variance / sd^2 on the fuel column; False (the default) leaves
              it unpenalised, as before. No history: the judge's prior uses no other race.
    """
    prior: Literal["circuit_history", "anchored"] = "circuit_history"
    practice: Literal["config", "anchored"] = "config"
    full_fit: bool = False


class PitLossModel(_Cfg):
    """Where a simulation's pit loss comes from (pit_loss.py, docs/eval.md "Pit loss and track
    status"). "snapshot" (the default, as before): snapshot.race.pit_loss_s and the effects'
    SC/VSC factors. "asof": pit_loss.estimate at the snapshot, from `prior` (configs/sim/
    pit_loss_prior.json, so config_hash covers it) and this race's stops seen by then, with
    its sd drawn per sample; the SC and VSC factors likewise. The world never takes "asof": it
    is given the race's whole-race value (Simulator(pit_loss_s=...)). The numbers below are
    set a priori, not tuned on any evaluation result."""

    source: Literal["snapshot", "asof"] = "snapshot"
    prior: dict = Field(default_factory=_load_pit_loss_prior)
    # the fallback chain's last tier, no earlier race at all: calibrate.DEFAULT_PIT_LOSS_S and an
    # sd wide enough to cover every whole-race value in the table (15 to 28.5 s)
    default_s: float = 22.0
    default_sd_s: float = 5.0
    # also pool every race in the FastF1 cache at run time (pit_loss.prior_pool), not the table only
    cache_lookup: bool = True
    # the world's whole-race value (eval.score.race_simulators): "median" pit_loss.whole_race, the
    # median green stop against the full fit; "replay_mean" the race's calibrate._pit_loss value
    world: Literal["median", "replay_mean"] = "median"
    min_green_stops: int = 3           # whole_race: below this many green stops, the prior
    same_circuit_sd_s: float = 1.5     # pit lane length and speed limit carry over
    min_cross_sd_s: float = 2.0
    stop_sd_floor_s: float = 2.5       # per-stop sd floor (stationary time spread)
    model_sd_s: float = 1.5            # per-stop estimand vs the whole-race definition
    plausible: tuple[float, float] = (0.5, 1.5)   # stops outside these multiples of the prior are dropped
    # SC / VSC factor prior: "config" the a priori default_*_factor (EventEffects' values, the
    # world's), "pooled" the table's stops (lost out of sample, reports/calibration/neutralised)
    factor_prior: Literal["config", "pooled"] = "config"
    default_sc_factor: float = 0.5
    default_vsc_factor: float = 0.6
    factor_sd: float = 0.2
    min_factor_stops: int = 5          # earlier races' stops needed for an as-of factor prior
    factor_prior_stops: float = 5.0    # the factor prior is worth this many stops
    factor_sd_floor: float = 0.15
def _cliff_history():
    from .cliff import load_history
    return load_history()


def _cliff_life_rows():
    from .cliff import load_life_ratio_rows
    return load_life_ratio_rows()


# the cost past a compound's support edge (sim/cliff.py, docs/eval.md "The wear cliff"): a lap
# o laps past it costs min(CLIFF_KAPPA o^2, CLIFF_CAP_S_PER_LAP) s more. kappa anywhere in
# 0.02-0.10 gives the same lap-0 stop counts on the development races, so it is 0.05 and not
# tuned; the cap is a dead tyre, not a wall, so one absurd plan cannot dominate a mean
CLIFF_KAPPA = 0.05
CLIFF_CAP_S_PER_LAP = 4.0
# belief tyre-life prior (sim/cliff.py life_prior) where no earlier race ran the compound: the
# oldest stint as a share of race distance, a priori (soft about two fifths, hard about two thirds)
PRIOR_LIFE_FRACTION = {"SOFT": 0.4, "MEDIUM": 0.55, "HARD": 0.7}
# its sd per tier of the fallback chain, a share of race distance, a priori and wider down the
# chain: the same circuit, earlier races anywhere, a neighbour's prior times the life ratio, the
# constant (the widest)
PRIOR_SD_FRACTION = {"circuit_history": 0.10, "history_fraction": 0.15, "life_ratio_as_of": 0.18,
                     "config_fraction": 0.20}


class PitStops(_Cfg):
    """Pit-stop durations (pit_loss.lane_deltas / team_delta, docs/eval.md "Pit-stop durations").
    The pit loss is one whole-race number; real stops vary by seconds and reorder close cars.

    replay  "charged" (as before): a whole-race replay of a real strategy charges every stop the
            pit loss; "actual": plus each stop's own lane time less the race's median green lane
            time (the timing data's pit entry to exit, hindsight; the fidelity tool's replays).
    team    every stop a simulation prices is the pit loss plus the snapshot car's team's typical
            stop: the median lane-time difference of the team's green stops, shrunk toward 0 by
            n / (n + team_prior_stops). The judge (fit_window "full") reads every stop of the race,
            a windowed fit the stops completed by the snapshot moment (pit_loss.snapshot_time).
    Off (the defaults): bit for bit as before. team_prior_stops 3, set a priori."""

    replay: Literal["charged", "actual"] = "charged"
    team: bool = False
    team_prior_stops: float = Field(default=3.0, ge=0)
class ScQueue(_Cfg):
    """The safety-car queue (sim/sc_queue.py, docs/eval.md "The safety-car queue"): at the end of
    each SC period the field's gaps close to `spacing_s` behind the leader in the order of the
    cars' time when it came out plus their stops under it. Off (the default): as before.
    spacing_s 0.5: the median real consecutive gap at the end of the calibration races' SC periods."""

    enabled: bool = False
    # "single": the engine places each simulated car in the queue among the other cars' race (the
    # judge: their recorded times; a windowed fit: the rows visible at the snapshot, projected
    # beyond them with gaps held), sc_queue.FieldQueue; "field": the fidelity tool's whole-field
    # replays queue every replayed car together (sc_queue.apply) and the engine does nothing
    mode: Literal["single", "field"] = "single"
    spacing_s: float = Field(default=0.5, ge=0)
    # the near-lapped rule (sc_queue.deployment_behind): a car more than this many laps behind the
    # leader when the safety car came out is held: not a car in the queue, it keeps its gap at the
    # deployment to the train's last car ahead of it (it was lapped on the road, and the wave-by came
    # too late for it to reach the back of the train).
    # None (the default): every car that finished on the lead lap joins, as before. 1.25: the cut
    # that best separates the cars that joined from those that did not on the recorded end-of-period
    # gaps of the calibration races' SC periods (fitted on the race data, not on a scorecard result)
    held_behind_laps: float | None = Field(default=None, gt=0)


class WearCliff(_Cfg):
    """Tyre life as physics (sim/cliff.py): past a compound's support edge E a lap costs
    min(kappa (age - E)^2, cap_s_per_lap) more. Both simulators start from the same historical
    prior (races strictly before this one); the world (fit_window full) extends it by the ages
    this race has lap data for (world_edge "history"), or uses the race alone ("race"); every
    other window places it from
    what is known at the snapshot: the ages seen so far as a lower bound and a pre-race
    tyre-life prior from earlier races (sim/cliff.py life_prior), sd prior_sd_fraction of race
    distance. `enabled: false` removes it, as the legacy R8 setting does (docs/rules.md R8)."""

    enabled: bool = True
    # the world's edge (sim/cliff.py world_cliffs): "history_only" (a: the historical prior both
    # simulators share, nothing from this race), "history" (b: that prior extended by the ages this
    # race has lap data for; the default until the fidelity scorecard chooses between a and b) or
    # "race" (c: the oldest age raced in this race, the earlier design)
    world_edge: Literal["history_only", "history", "race"] = "history"
    # the historical prior (sim/cliff.py history_prior): "tyre_history" (src/sim/tyre_history.py,
    # the shared history model) or "stub" (cliff.life_prior, the earlier as-of stub)
    prior_source: Literal["stub", "tyre_history"] = "tyre_history"
    kappa: float = Field(default=CLIFF_KAPPA, ge=0)
    cap_s_per_lap: float = Field(default=CLIFF_CAP_S_PER_LAP, ge=0)
    # belief prior on tyre life as a share of race distance, when no earlier race has the compound
    prior_life_fraction: dict[str, float] = Field(default_factory=lambda: dict(PRIOR_LIFE_FRACTION))
    prior_sd_fraction: dict[str, float] = Field(default_factory=lambda: dict(PRIOR_SD_FRACTION))
    # also read every cached race strictly before the race (held-out ones never) at run time, so
    # newly cached seasons enter the prior without a code change (sim/cliff.py history_for)
    scan_cache: bool = True
    # races whose tyre ages say nothing about tyre life (a mandated stint cap)
    life_excluded: list[str] = Field(default_factory=lambda: ["2025_23_qatar"])
    history: dict = Field(default_factory=_cliff_history)          # configs/sim/tyre_life_history.json
    life_ratio_rows: list = Field(default_factory=_cliff_life_rows)  # compound_steps.json per_race


class SimConfig(_Cfg):
    degradation: ComponentSpec = ComponentSpec(name="linear_fuel")
    events: ComponentSpec = ComponentSpec(name="none")
    effects: EventEffects = Field(default_factory=EventEffects)
    hazard: ComponentSpec = ComponentSpec(name="none")
    hazard_effects: HazardEffects = Field(default_factory=HazardEffects)
    rules: list[str] = Field(default_factory=lambda: [
        "first_stint_matches", "lap_sum", "known_compounds", "min_stint", "two_compounds", "inventory",
        "stint_limit",
    ])
    inventory_sources: list[str] = Field(default_factory=lambda: ["observed", "inferred", "assumed"])
    min_samples: int = 5      # clean laps needed before a compound is modelled
    n_samples: int = 1
    # Which laps the degradation model is fitted on, per snapshot (Simulator.fitted_for):
    #   full         every race lap (hindsight, the world)
    #   to_snapshot  race laps up to snapshot.laps_completed, shrunk toward the pre_race fit
    #   pre_race     practice long runs only
    fit_window: Literal["full", "to_snapshot", "pre_race"] = "full"
    exclude_snapshot_driver: bool = False   # leave the snapshot driver's laps out of the fit
    # what a windowed fit may read of the race frame at the snapshot (PolicyView); left out of
    # config_hash while at its default
    policy_view: PolicyView = Field(default_factory=PolicyView)
    # a windowed fit reads calibration_error_s (model_report.calibration_error) as of the race:
    # NaN unless every race of the midrace summary started before it (docs/eval.md "Information
    # contract"); False reads it whatever the race (legacy). Left out of config_hash at its default.
    cross_race_as_of: bool = True
    # fit on free-air rows only: every degradation fit (all windows), the hazard fit and
    # ref_lap_s use the clean rows with data.lap_frame's free_air True; practice rows pass
    # (Simulator._fit). ref_lap_s becomes the free-air median, while sc_lap_factor (1.36) was
    # measured against the unfiltered one; the cross-race factor fit measures both.
    fit_free_air_only: bool = False
    # practice cannot identify the race fuel term; linear_fuel uses this when no race laps fit
    # it, and shrinks toward it under to_snapshot
    pre_race_fuel_slope_s_per_lap: float = -0.045
    # fuel anchoring (FuelAnchor); left out of config_hash at its default
    fuel_anchor: FuelAnchor = Field(default_factory=FuelAnchor)
    # practice-to-race base level, s/lap, added to every practice-derived compound base in the
    # pre_race prior (Spain: full-fit base 85.22 less practice 84.18, rounded). A config prior
    # for now (fit_report base_delta_source "prior"); the first factor SPEC H's calibrator
    # will predict per race. to_snapshot shrinks it with the base_s tau.
    pre_race_base_delta_s: float = 1.0
    # to_snapshot prior standard deviations around the pre_race fit (see degradation.py). Fuel
    # is tight: burn is near a physical constant per circuit, and the wear/fuel split is not
    # identified while every car is in its first stint.
    shrinkage: dict[str, float] = Field(default_factory=lambda: {
        "base_s": 0.5, "wear_s_per_lap": 0.03, "fuel_s_per_lap": 0.008,
        "hinge_s_per_lap": HINGE_TAU_S_PER_LAP,   # linear_fuel_hinge only
    })
    # linear_fuel_hinge: tyre age where a compound's hinge starts, hinge_knot_default for the rest
    hinge_knots: dict[str, int] = Field(default_factory=dict)
    hinge_knot_default: int = HINGE_KNOT
    # Calibrator (calibrators.py) for the pre_race and to_snapshot windows: "shrinkage" (the
    # to_snapshot fit toward the practice prior, the default), "none" (physics plus config
    # values, held), or a learned class. The full window never calls one.
    calibrator: ComponentSpec = ComponentSpec(name="shrinkage")
    # calibration cadence N: recalibrate at multiples of N laps (1 = every decision point,
    # None = pre-race only), and at an SC or VSC start or end, rain onset, or a track
    # temperature change above recalibrate_temp_delta_c since the last calibration point
    calibration_every: int | None = Field(default=1, ge=1)
    recalibrate_temp_delta_c: float = Field(default=3.0, gt=0)
    long_run: LongRun = Field(default_factory=LongRun)   # practice long-run test, for data.practice_frame
    # tyre class per compound (SLICK, INTERMEDIATE, WET). Machinery for wet-weather rule
    # waivers and track-state factors; no wet behaviour is modelled.
    compound_classes: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_COMPOUND_CLASSES))
    track: ComponentSpec = ComponentSpec(name="dry")   # track-state transform, see track.py
    # traffic model, see traffic.py: "none" (default) or "ghost", the recorded field replayed
    # as ghost cars (source "recorded", world only) or projected from the laps so far ("projected", belief)
    traffic: ComponentSpec = ComponentSpec(name="none")
    # residual lap noise: "none" (default) or "residual", per sample an independent
    # N(0, residual_s^2) on every remaining green lap, residual_s from the degradation
    # fit_report, one vector per sample shared by every plan (engine LAP_NOISE_STREAM).
    # The degradation parameter draw is the model's `draw` param (degradation.py).
    # "persistent" (docs/eval.md "Honest uncertainty"): the residuals are not independent
    # (lag-1 autocorrelation 0.78-0.94), so each sample draws one offset per plan stint, held
    # over the stint, plus the lap-to-lap part iid, both sds from the fit's residual_components
    # (engine._lap_noise_offsets, LAP_OFFSET_STREAM).
    lap_noise: Literal["none", "residual", "persistent"] = "none"
    # what optimize_batch (the belief optimizer, the rule baselines' search and the tools'
    # continuations) ranks candidates on: "draws" (the default, as before) the mean over the
    # posterior parameter draws and lap noise; "point" the point estimate (no parameter draw, no
    # lap noise; events and pit-loss draws kept), Simulator.point_estimate. The tools' reported
    # intervals still come from the draws. Left out of config_hash at its default.
    plan_on: Literal["draws", "point"] = "draws"
    # the belief's wear-cliff edge per sample (sim/cliff.py Cliff.sample_ages): "draw" (the
    # default, as before) max(seen so far, prior + sd z); "point" max(seen so far, prior), the
    # edge the judge's history_only prior uses. The cost past the edge is convex, so drawing the
    # edge raises the expected cost of every long stint (docs/eval.md "Belief priors"). Left out
    # of config_hash at its default.
    belief_cliff_edge: Literal["draw", "point"] = "draw"
    # R8 stint limit, RETIRED (docs/rules.md R8): a stint could end at most this many laps past
    # the oldest tyre age any car ran on its compound in the whole race, a hindsight rule handed
    # to every policy. None (the default) is off. The legacy setting that reproduces the results
    # before its retirement is stint_limit_margin_laps 3 with wear_cliff.enabled false
    # (configs/sim/legacy/); then the "stint_limit" rule binds simulate_batch,
    # optimize_batch's candidate filter and the agent's tools as before.
    stint_limit_margin_laps: int | None = Field(default=None, ge=0)
    # R8: the limit is the age at least this many cars reached on the compound (the rank-th
    # largest per-car maximum); 1 is the oldest tyre any car ran
    stint_limit_rank: int = Field(default=1, ge=1)
    # unraced dry compounds priced by proxy (proxy.py); compound_proxy.enabled False restores the
    # simulator as it was before proxying, bit for bit
    compound_proxy: CompoundProxy = Field(default_factory=CompoundProxy)
    # pit loss as of the snapshot (PitLossModel); the default reads the snapshot's, as before
    pit_loss: PitLossModel = Field(default_factory=PitLossModel)
    # a compound with fewer clean race rows than this in a fit's race rows is left to the compound
    # proxy rather than fitted from a handful of laps (Simulator._thin_dropped); None (the default)
    # fits every compound with min_samples rows, as before
    min_race_rows_to_fit: int | None = Field(default=None, ge=1)
    # over-long stints priced by a wear cliff (sim/cliff.py), the physics that replaced R8
    wear_cliff: WearCliff = Field(default_factory=WearCliff)
    # pit-stop durations (PitStops); left out of config_hash at its default
    pit_stops: PitStops = Field(default_factory=PitStops)
    # the safety-car queue in whole-field replays (ScQueue); left out of config_hash at its default
    sc_queue: ScQueue = Field(default_factory=ScQueue)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            if str(path).endswith((".yaml", ".yml")):
                import yaml
                return cls.model_validate(yaml.safe_load(f))
            return cls.model_validate(json.load(f))


# The eval world's wear fit (score.race_simulators' default world, docs/eval.md "The world's wear
# fit"): linear_fuel with driver effects on the race rows. The fidelity scorecard (src/sim/
# fidelity.py, six races) finds driver effects the one factor that improves its primary measure
# (close-pair finishing order) and the hinge adds nothing (DE + hinge slightly worse than DE
# alone); the cross-race G rule on the calibration races does not tell the two apart (DE against
# DE + hinge: +0.26 [-0.08, 0.62] s), so the simpler one is the default. Both are named world
# overlays for the scorecard's ablation: configs/sim/world_de.yaml and world_de_hinge.yaml. The
# SimConfig default stays linear_fuel, the comparator the cross-race arms are overlays on.
WORLD_DEGRADATION = {"name": "linear_fuel", "params": {"driver_effects": "race"}}
# The eval world's tyre-life edge (WearCliff.world_edge): "history_only", the historical prior both
# simulators share and nothing from this race. Chosen by the fidelity scorecard (docs/eval.md "What
# the ablation says"): close-pair order +0.025 [+0.000, +0.060] against "history" on 18 races, and
# no hindsight. The SimConfig default stays "history"; the eval world sets it here. The earlier
# world (edge "history") is configs/sim/legacy/world_pre_scorecard.yaml.
WORLD_EDGE = "history_only"


def world_default(config: "SimConfig") -> "SimConfig":
    """`config` with the eval world's wear fit (WORLD_DEGRADATION) and tyre-life edge (WORLD_EDGE)."""
    return config.model_copy(update={"degradation": ComponentSpec(**WORLD_DEGRADATION),
                                     "wear_cliff": config.wear_cliff.model_copy(update={"world_edge": WORLD_EDGE})})


def compound_class(config, compound) -> str:
    """The tyre class of a compound. An unknown compound counts as SLICK (with a warning);
    known_compounds still rejects compounds the degradation model has not seen."""
    classes = DEFAULT_COMPOUND_CLASSES if config is None else config.compound_classes
    if compound not in classes:
        log.warning("compound %r has no class, treating it as SLICK", compound)
        return "SLICK"
    return classes[compound]


def class_names(config) -> list[str]:
    """Class labels in a fixed order (SLICK first), so engine class codes depend on the
    config only and never on the plans scored."""
    classes = DEFAULT_COMPOUND_CLASSES if config is None else config.compound_classes
    return ["SLICK", *sorted(set(classes.values()) - {"SLICK"})]
