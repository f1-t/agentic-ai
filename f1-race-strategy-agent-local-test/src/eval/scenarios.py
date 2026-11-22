"""Scenario sets: fixed, versioned lists of decision points the eval scores.

A scenario is one (race, driver, lap) snapshot. A set is sampled once from the snapshot
builder and written to json with every snapshot in full, so a later change to the builder,
the cache or the frame filter does not move a published set. Bump the version whenever the
sampling, the builder or the race list changes; never edit a published file.

    python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json

Sampling, per race, under sampling["rule"] (docs/eval.md "Information contract"):

  as_of   (the default for new sets) every driver still running at the decision lap, at
          decision_laps_as_of (every 5th lap; optionally the laps a neutralisation is in progress or
          has just ended, read off the driver's own record up to that lap), snapshots by
          calibration.snapshot_as_of (scheduled distance, nominated compounds, records
          completed by the decision moment only). A driver who later retired or was lapped
          is sampled like any other; the world scores every plan over the scheduled distance.
  legacy  every set committed before the leak fixes (v1 to dev6, d2 to d4; their sampling
          carries no "rule"): drivers whose laps run 1..n without a gap and who finished at
          most `max_laps_down` laps behind the race length (DNFs are out), at the decision
          laps of calibration.default_laps_k (every 5th lap plus the laps either side of each
          SC and VSC period), snapshots by calibration.snapshot_at (the driver's own lap count
          as the race length). Both the selection and the race length read the future; the
          sets stay loadable and scoreable as they were.

Either way the laps leave room for one more stint, optionally subsampled to `max_per_race`
with a seeded rng.

The registered eval split (docs/eval.md "Development and report sets"). data/eval_set.json,
written by the data team's register_eval_set, lists the races held out for reporting:

    {"races": ["2022_05_miami", ...], ...}      (or a bare list of race ids)

While it exists, build_set refuses any race outside it, records the file's sha256 in
sampling["eval_set_digest"] and marks the set development=False: a report set. Without it a
set is a development set (development=True), which v1 is: sampled before the split, from a
race we tuned on. build_set(..., development=True) (`scenarios --development`) samples a
development set while the split exists, a debug set of any race: no refusal, and the digest
is still recorded so the set shows the split existed. Report numbers come from sets with
development=False only.

A race in CALIBRATION_ONLY is refused by build_set in every mode, development included: it
calibrates the simulator but is never sampled into a scenario set (docs/eval.md "Sprint
weekends").
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import RaceParams, Snapshot
from ..sim import calibration
from ..sim.data import race_stints
from ..sim.race_ids import canonical_race_id

SCHEMA_VERSION = 1
EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"
# The data team's snapshot builder replaces this; it has the same signature.
SnapshotBuilder = Callable[[pd.DataFrame, RaceParams, str, int], Snapshot]
DEFAULT_BUILDER = "src.sim.calibration.snapshot_at"   # the legacy builder, and a set's default field
AS_OF_BUILDER = "src.sim.calibration.snapshot_as_of"   # the as_of rule's builder (docs/eval.md "Information contract")


# Snapshot fields added to the contract after sets were published (SPEC 0014 I). The digest
# leaves each out while it holds its default, so a set saved before the field existed keeps
# its digest, and a set that fills the field hashes it.
_ADDED_FIELDS = {("snapshot", "recent_lap_times_s"): [], ("snapshot", "context"): {},
                 ("snapshot", "weather", "rain_probability"): None, ("snapshot", "field", "nearby"): [],
                 ("stratum",): None, ("weight",): None}


def _digest_view(item: dict) -> dict:
    for path, default in _ADDED_FIELDS.items():
        parent = item
        for key in path[:-1]:
            parent = parent.get(key, {})
        if path[-1] in parent and parent[path[-1]] == default:
            del parent[path[-1]]
    return item


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str                     # <race_id>:<driver>:<lap:02d>
    race_id: str
    driver: str
    lap: int = Field(ge=0)               # laps completed at the decision point
    snapshot: Snapshot
    actual_stints: list[tuple[str, int]]  # the driver's full-race stints, for the actual-strategy policy
    # stratified sets only (sample_stratified): the scorer-side stratum and the inclusion weight
    # (stratum pool size / sampled). Never shown to a policy: score.policy_scenario drops both.
    stratum: str | None = None
    weight: float | None = None


class ScenarioSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    version: str
    builder: str = DEFAULT_BUILDER
    sampling: dict = Field(default_factory=dict)
    sim_code_version: str = ""
    digest: str = ""                     # sha256 of the items, set by save(), checked by load()
    # True: sampled without the registered eval split (tuning and debugging; v1 is one).
    # False: sampled from the eval races only, the set the report numbers come from.
    development: bool = True
    items: list[Scenario] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)   # {race_id, driver, lap, reason}

    def compute_digest(self) -> str:
        payload = json.dumps([_digest_view(s.model_dump(mode="json")) for s in self.items], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.digest = self.compute_digest()
        path.write_text(self.model_dump_json(indent=1) + "\n")
        return path

    @property
    def kind(self) -> str:
        """"development set" or "report set", for the score CLI's one-line notice."""
        return "development set" if self.development else "report set"

    @classmethod
    def load(cls, path) -> ScenarioSet:
        out = cls.model_validate_json(Path(path).read_text())
        if out.digest and out.digest != out.compute_digest():
            raise ValueError(f"{path}: items do not match the recorded digest; scenario sets are frozen, "
                             "build a new version instead of editing one")
        return out

    def chunk(self, index: int, size: int) -> list[Scenario]:
        return self.items[index * size:(index + 1) * size]


def eligible_drivers(frame: pd.DataFrame, total_laps: int, max_laps_down: int = 2) -> list[str]:
    """Drivers with laps 1..n and no gap, n at least total_laps - max_laps_down."""
    out = []
    for driver, g in frame.groupby("driver", sort=True):
        laps = sorted(int(x) for x in g["lap"])
        if laps == list(range(1, len(laps) + 1)) and len(laps) >= total_laps - max_laps_down:
            out.append(str(driver))
    return out


def decision_laps(frame: pd.DataFrame, race: RaceParams, driver_laps: int) -> list[int]:
    """calibration.default_laps_k, kept to laps that leave at least one minimum stint to run."""
    return [k for k in calibration.default_laps_k(frame, race.total_laps)
            if k <= driver_laps - race.min_stint_laps]


GRID_EVERY = 5          # as_of decision laps: every 5th lap from 0 (the stratified pool: every 2nd)


def _neutralised(status) -> bool:
    s = str(status)
    return "4" in s or "6" in s or "7" in s


def as_of_pool(frame: pd.DataFrame, race: RaceParams, every: int) -> list[tuple[str, int]]:
    """The one candidate pool of the as_of and stratified rules: (driver, L) for every L from 0
    to total_laps - min_stint_laps in steps of `every`, for the drivers running at L (their laps
    1..L all present; lap 1 at L = 0). Reads no lap after L, and the scheduled race length, not
    the driver's final lap count: whether a car later retired or was lapped is not read."""
    out = []
    for k in range(0, race.total_laps - race.min_stint_laps + 1, every):
        seen = frame[frame["lap"] <= max(k, 1)]
        for driver, g in seen.groupby("driver", sort=True):
            if sorted(int(x) for x in g["lap"]) == list(range(1, max(k, 1) + 1)):
                out.append((str(driver), k))
    return out


def decision_laps_as_of(frame: pd.DataFrame, race: RaceParams, driver: str, neutralised: bool = False,
                        every: int = GRID_EVERY) -> list[int]:
    """The as_of rule's decision laps for one driver: its laps in as_of_pool (every `every`
    laps). With `neutralised`, also every lap L it was running at whose own record shows a
    neutralisation in progress (SC or VSC in its lap-L track status) or just ended (lap L - 1
    neutralised, lap L not), within the same cap. Off by default: such a lap is chosen at the
    driver's own moment, which for a car behind comes after other cars' moments at the same
    lap number, so the set of points up to a lap would depend on later records."""
    g = frame[frame["driver"] == driver]
    laps = {k for _, k in as_of_pool(g, race, every)}
    if neutralised:
        status = {int(lap): st for lap, st in zip(g["lap"], g["track_status"])}
        running = 0   # laps 1..running recorded without a gap
        while running + 1 in status:
            running += 1
        for lap, st in status.items():
            if (_neutralised(st) or (lap > 1 and _neutralised(status.get(lap - 1, "")))) and lap <= running:
                laps.add(lap)
        laps = {k for k in laps if k <= race.total_laps - race.min_stint_laps}
    return sorted(laps)


# "as_of" (default) and "legacy" are sample_race's; "stratified" is sample_stratified's
SAMPLING_RULES = ("as_of", "stratified", "legacy")


def sample_race(frame: pd.DataFrame, race: RaceParams, *, builder: SnapshotBuilder | None = None,
                drivers: list[str] | None = None, max_laps_down: int = 2,
                max_per_race: int | None = None, seed: int = 0,
                rule: str = "as_of", neutralised: bool = False) -> tuple[list[Scenario], list[dict]]:
    """(scenarios, skipped) for one race, in (driver, lap) order.

    rule "as_of" (the default for new sets): every driver on the grid, at
    decision_laps_as_of, built by calibration.snapshot_as_of unless `builder` says otherwise;
    max_laps_down is not read. rule "legacy": the rule every set up to dev6 and d4 was sampled
    under, which reads the future: eligible_drivers (finishers at most max_laps_down laps down)
    at decision_laps (default_laps_k: the lap before each SC and VSC period too), built by
    calibration.snapshot_at by default."""
    if rule not in ("as_of", "legacy"):
        raise ValueError(f"sampling rule must be one of {SAMPLING_RULES} (stratified: sample_stratified), "
                         f"got {rule!r}")
    race_id = race.race_id
    if builder is None:
        builder = calibration.snapshot_as_of if rule == "as_of" else calibration.snapshot_at
    if rule == "as_of":
        pool = sorted(str(d) for d in frame["driver"].dropna().unique())
    else:
        pool = eligible_drivers(frame, race.total_laps, max_laps_down)
    if drivers is not None:
        pool = [d for d in pool if d in set(drivers)]
    points = []
    for driver in pool:
        if rule == "as_of":
            points += [(driver, k) for k in decision_laps_as_of(frame, race, driver, neutralised)]
        else:
            n = int((frame["driver"] == driver).sum())
            points += [(driver, k) for k in decision_laps(frame, race, n)]
    if max_per_race is not None and len(points) > max_per_race:
        points = sorted(random.Random(f"{seed}:{race_id}").sample(points, max_per_race))
    items, skipped = [], []
    for driver, k in points:
        try:
            snapshot = builder(frame, race, driver, k)
        except ValueError as exc:
            skipped.append({"race_id": race_id, "driver": driver, "lap": k, "reason": str(exc)})
            continue
        actual = [(str(c), int(n)) for c, n, _, _ in race_stints(frame, driver)]
        items.append(Scenario(scenario_id=f"{race_id}:{driver}:{k:02d}", race_id=race_id, driver=driver,
                              lap=k, snapshot=snapshot, actual_stints=actual))
    return items, skipped


# --- the stratified (decision-critical) sampler: rule "stratified" -------------------------------
STRATA = ("pit_opt", "close", "neutral", "window", "rest")   # assigned in this order
DEFAULT_QUOTAS = {"pit_opt": 7, "close": 6, "neutral": 4, "window": 4, "rest": 4}   # per race, 25
POOL_EVERY = 2            # the pool: every 2nd lap
CLOSE_GAP_S = 3.0         # close: |best pit-now - best stay-out| at most this ...
CLOSE_SPREAD_S = 10.0     # ... and the worst searched plan at least this far behind the reference
WINDOW_LAPS = 2           # window: some car pitted on laps L - 2 .. L
AFTER_LAPS = 2            # neutral: under SC/VSC at L, or one ended within the 2 laps before L


def stratified_pool(frame: pd.DataFrame, race: RaceParams, every: int = POOL_EVERY) -> list[tuple[str, int]]:
    """as_of_pool every `every` (2) laps: the stratified rule's candidates, the as_of rule's pool
    at a finer step."""
    return as_of_pool(frame, race, every)


# What the stratified rule reads beyond the as_of pool is scorer-side and never reaches a policy:
# the strata (record_stratum, reference_stratum: the world's hindsight reference) and the
# weights choose WHICH candidates are scored and how they are weighted; each scored snapshot is
# still built by the as_of builder from the records up to its own moment, and
# score.policy_scenario strips stratum and weight before a policy is called. The selection is
# conditioned on the future by design (decision-critical points), corrected by the weights.
def record_stratum(frame: pd.DataFrame, driver: str, k: int) -> str | None:
    """"neutral" or "window" from the race's laps up to L only, else None: neutral when the
    driver's lap L is under SC or VSC or one of the AFTER_LAPS laps before it was (and L is
    green); window when some car pitted on laps L - WINDOW_LAPS .. L. Scorer-side (above)."""
    if k == 0:
        return None
    seen = frame[frame["lap"] <= k]
    own = seen[seen["driver"] == driver].set_index("lap")["track_status"].astype(str)

    def neutral(lap):
        s = own.get(lap, "1")
        return "4" in s or "6" in s or "7" in s
    if neutral(k) or any(neutral(j) for j in range(k - AFTER_LAPS, k)):
        return "neutral"
    recent = seen[(seen["lap"] >= k - WINDOW_LAPS) & (seen["lap"] >= 1)]
    return "window" if bool(recent["pit_in"].astype(bool).any()) else None


def reference_stratum(ref, snapshot) -> str | None:
    """"pit_opt" (the world's reference pits at the end of the next lap) or "close" (not
    pit_opt; the best pit-now and best stay-out plans within CLOSE_GAP_S of each other, and the
    worst searched plan at least CLOSE_SPREAD_S behind the reference: a call that matters and is
    hard), else None. `ref` is eval.reference.compute_reference(..., keep_all=True) on the world.
    Scorer-side: a policy never sees the reference, and the stratum only chooses which snapshots
    are scored and with what weight."""
    from .reference import near_ties
    if ref is None or ref.status != "ok":
        return None
    if ref.pit_now_opt:
        return "pit_opt"
    gap = near_ties(ref, snapshot)["pit_now_gap_s"]
    spread = float(np.max(ref.plan_gap_s)) if len(ref.plan_gap_s) else 0.0
    return "close" if np.isfinite(gap) and abs(gap) <= CLOSE_GAP_S and spread >= CLOSE_SPREAD_S else None


def sample_stratified(frame: pd.DataFrame, race: RaceParams, world, *, builder: SnapshotBuilder = calibration.snapshot_as_of,
                      quotas: dict | None = None, seed: int = 0, n_samples: int | None = None, max_stops: int = 2,
                      every: int = POOL_EVERY, drivers: list[str] | None = None) -> tuple[list[Scenario], list[dict]]:
    """(scenarios, skipped) of one race under the stratified rule: every stratified_pool point
    gets the first stratum in STRATA that applies (reference_stratum on the `world`'s reference,
    then record_stratum, else "rest"); each stratum is sampled down to its quota (DEFAULT_QUOTAS)
    with a seeded rng, a shortfall going to "rest"; every item carries its stratum and weight
    (stratum pool size / sampled), so a pooled estimate reweights to the uniform pool. Depends
    on the race record and the world reference only, never on a policy."""
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    race_id = race.race_id
    pool = [(d, k) for d, k in stratified_pool(frame, race, every) if drivers is None or d in set(drivers)]
    by, skipped = {s: [] for s in STRATA}, []
    from .reference import compute_reference
    for driver, k in pool:
        try:
            snapshot = builder(frame, race, driver, k)
        except ValueError as exc:
            skipped.append({"race_id": race_id, "driver": driver, "lap": k, "reason": str(exc)})
            continue
        actual = [(str(c), int(n)) for c, n, _, _ in race_stints(frame, driver)]
        item = Scenario(scenario_id=f"{race_id}:{driver}:{k:02d}", race_id=race_id, driver=driver, lap=k,
                        snapshot=snapshot, actual_stints=actual)
        ref = compute_reference(item, world, seed=seed, n_samples=n_samples, max_stops=max_stops, keep_all=True)
        stratum = reference_stratum(ref, snapshot) or record_stratum(frame, driver, k) or "rest"
        by[stratum].append(item)
    items, spare = [], 0
    for s in STRATA:
        n = quotas.get(s, 0) + (spare if s == "rest" else 0)
        got = by[s] if len(by[s]) <= n else random.Random(f"{seed}:{race_id}:{s}").sample(by[s], n)
        spare += 0 if s == "rest" else max(0, n - len(by[s]))
        items += [x.model_copy(update={"stratum": s, "weight": len(by[s]) / len(got)}) for x in got]
    return sorted(items, key=lambda x: (x.driver, x.lap)), skipped


class OutsideEvalSet(ValueError):
    """A race asked for while the eval split is registered is not in it."""


# Races used for calibration only, never sampled into a scenario set, with the reason.
CALIBRATION_ONLY = {
    "2025_23_qatar": "a mandated maximum of 25 laps per tyre set, which no rule in R1-R8 models",
}


class CalibrationOnly(ValueError):
    """A race asked for is in CALIBRATION_ONLY."""


def load_eval_set(path=None) -> tuple[list[str], str] | None:
    """(canonical race ids, sha256 of the file) of the registered eval split, None when the
    file does not exist. `path` defaults to data/eval_set.json."""
    path = Path(EVAL_SET if path is None else path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    data = json.loads(raw)
    races = data if isinstance(data, list) else data["races"]
    return [canonical_race_id(str(r)) for r in races], hashlib.sha256(raw).hexdigest()


def check_schedule_ids(race_ids) -> None:
    """Every race id is the one its season's cached schedule gives it
    (race_ids.make_race_id_from_schedule, through sim.data.schedule_race_id), so scenario ids
    name races as real-run discovery and the data layout do. A season without a cached
    schedule is not checked. Raises ValueError naming the expected id."""
    from ..sim.data import schedule_race_id
    from ..sim.race_ids import parse_race_id
    for r in race_ids:
        rid = canonical_race_id(str(r))
        season, rnd, _ = parse_race_id(rid)
        expected = schedule_race_id(season, rnd)
        if expected is not None and expected != rid:
            raise ValueError(f"race id {rid!r}: the {season} schedule names round {rnd} {expected!r}")


def build_set(race_ids: list[str], version: str, *, profile=None, builder: SnapshotBuilder | None = None,
              builder_name: str | None = None, eval_set=None, development: bool = False,
              rule: str = "as_of", **sampling) -> ScenarioSet:
    """Sample every race in `race_ids` from the committed cache (sim.jobs.load_race).

    `eval_set` is the registered split's file, data/eval_set.json when None. When it exists,
    a race outside it raises OutsideEvalSet before anything is sampled, and the set is a
    report set (development False) stamped with sampling["eval_set_digest"]. Without it the
    set is a development set. `development` True asks for a development set while the split
    exists: any race is sampled and the digest is still recorded. A CALIBRATION_ONLY race
    raises CalibrationOnly in either mode, before anything is sampled. `rule` (SAMPLING_RULES,
    "as_of" by default; "stratified" runs sample_stratified on the world) is recorded as
    sampling["rule"]; the builder defaults to AS_OF_BUILDER (DEFAULT_BUILDER under "legacy"),
    and its name is recorded as the set's builder."""
    from ..sim import jobs
    from ..sim.compute import load_profile

    banned = [r for r in map(canonical_race_id, race_ids) if r in CALIBRATION_ONLY]
    if banned:
        raise CalibrationOnly("; ".join(f"{r} is used for calibration only and never sampled into a scenario "
                                        f"set: {CALIBRATION_ONLY[r]}" for r in banned))
    registered = load_eval_set(eval_set)
    split = {}
    if registered is not None:
        allowed, digest = registered
        outside = [canonical_race_id(r) for r in race_ids if canonical_race_id(r) not in set(allowed)]
        if outside and not development:
            raise OutsideEvalSet(f"races {outside} are not in the registered eval set ({len(allowed)} races, "
                                 f"sha256 {digest[:12]}); a set sampled under the split takes eval races only")
        split = {"eval_set_digest": digest}
    profile = profile or load_profile()
    check_schedule_ids(race_ids)
    items, skipped = [], []
    if rule not in SAMPLING_RULES:
        raise ValueError(f"sampling rule must be one of {SAMPLING_RULES}, got {rule!r}")
    if builder is None:
        builder = calibration.snapshot_at if rule == "legacy" else calibration.snapshot_as_of
        builder_name = builder_name or (DEFAULT_BUILDER if rule == "legacy" else AS_OF_BUILDER)
    world_for = None
    if rule == "stratified":   # the scorer's world stratifies (sample_stratified)
        from .score import race_simulators
        world_for = race_simulators(profile)
    for race_id in race_ids:
        data = jobs.load_race(race_id, profile)
        if world_for is not None:
            kw = {k: v for k, v in sampling.items() if k in ("quotas", "seed", "drivers", "max_stops")}
            got, miss = sample_stratified(data.frame, data.race, world_for(data.race_id, "world"), builder=builder,
                                          n_samples=profile.sim.n_samples, **kw)
        else:
            got, miss = sample_race(data.frame, data.race, builder=builder, rule=rule, **sampling)
        items += got
        skipped += miss
    return ScenarioSet(version=version, builder=builder_name or DEFAULT_BUILDER,
                       sampling={"races": [canonical_race_id(r) for r in race_ids], "rule": rule, **sampling,
                                 **split},
                       sim_code_version=calibration.SIM_CODE_VERSION, items=items, skipped=skipped,
                       development=development or registered is None)
