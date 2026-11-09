"""Scenario sets: fixed, versioned lists of decision points the eval scores.

A scenario is one (race, driver, lap) snapshot. A set is sampled once from the snapshot
builder and written to json with every snapshot in full, so a later change to the builder,
the cache or the frame filter does not move a published set. Bump the version whenever the
sampling, the builder or the race list changes; never edit a published file.

    python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json

Sampling, per race: drivers whose laps run 1..n without a gap and who finished at most
`max_laps_down` laps behind the race length (DNFs are out), at the decision laps of
calibration.default_laps_k (every 5th lap plus the laps either side of each SC and VSC
period) that leave room for one more stint, optionally subsampled to `max_per_race` with a
seeded rng.

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
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path

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
DEFAULT_BUILDER = "src.sim.calibration.snapshot_at"


# Snapshot fields added to the contract after sets were published (SPEC 0014 I). The digest
# leaves each out while it holds its default, so a set saved before the field existed keeps
# its digest, and a set that fills the field hashes it.
_ADDED_FIELDS = {("snapshot", "recent_lap_times_s"): [], ("snapshot", "context"): {},
                 ("snapshot", "weather", "rain_probability"): None, ("snapshot", "field", "nearby"): []}


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


def sample_race(frame: pd.DataFrame, race: RaceParams, *, builder: SnapshotBuilder = calibration.snapshot_at,
                drivers: list[str] | None = None, max_laps_down: int = 2,
                max_per_race: int | None = None, seed: int = 0) -> tuple[list[Scenario], list[dict]]:
    """(scenarios, skipped) for one race, in (driver, lap) order."""
    race_id = race.race_id
    pool = eligible_drivers(frame, race.total_laps, max_laps_down)
    if drivers is not None:
        pool = [d for d in pool if d in set(drivers)]
    points = []
    for driver in pool:
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


class OutsideEvalSet(ValueError):
    """A race asked for while the eval split is registered is not in it."""


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


def build_set(race_ids: list[str], version: str, *, profile=None, builder: SnapshotBuilder = calibration.snapshot_at,
              builder_name: str = DEFAULT_BUILDER, eval_set=None, development: bool = False,
              **sampling) -> ScenarioSet:
    """Sample every race in `race_ids` from the committed cache (sim.jobs.load_race).

    `eval_set` is the registered split's file, data/eval_set.json when None. When it exists,
    a race outside it raises OutsideEvalSet before anything is sampled, and the set is a
    report set (development False) stamped with sampling["eval_set_digest"]. Without it the
    set is a development set. `development` True asks for a development set while the split
    exists: any race is sampled and the digest is still recorded."""
    from ..sim import jobs
    from ..sim.compute import load_profile

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
    items, skipped = [], []
    for race_id in race_ids:
        data = jobs.load_race(race_id, profile)
        got, miss = sample_race(data.frame, data.race, builder=builder, **sampling)
        items += got
        skipped += miss
    return ScenarioSet(version=version, builder=builder_name,
                       sampling={"races": [canonical_race_id(r) for r in race_ids], **sampling, **split},
                       sim_code_version=calibration.SIM_CODE_VERSION, items=items, skipped=skipped,
                       development=development or registered is None)
