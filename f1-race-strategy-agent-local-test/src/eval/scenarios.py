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
# The data team's snapshot builder replaces this; it has the same signature.
SnapshotBuilder = Callable[[pd.DataFrame, RaceParams, str, int], Snapshot]
DEFAULT_BUILDER = "src.sim.calibration.snapshot_at"


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
    items: list[Scenario] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)   # {race_id, driver, lap, reason}

    def compute_digest(self) -> str:
        payload = json.dumps([s.model_dump(mode="json") for s in self.items], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.digest = self.compute_digest()
        path.write_text(self.model_dump_json(indent=1) + "\n")
        return path

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


def build_set(race_ids: list[str], version: str, *, profile=None, builder: SnapshotBuilder = calibration.snapshot_at,
              builder_name: str = DEFAULT_BUILDER, **sampling) -> ScenarioSet:
    """Sample every race in `race_ids` from the committed cache (sim.jobs.load_race)."""
    from ..sim import jobs
    from ..sim.compute import load_profile

    profile = profile or load_profile()
    items, skipped = [], []
    for race_id in race_ids:
        data = jobs.load_race(race_id, profile)
        got, miss = sample_race(data.frame, data.race, builder=builder, **sampling)
        items += got
        skipped += miss
    return ScenarioSet(version=version, builder=builder_name,
                       sampling={"races": [canonical_race_id(r) for r in race_ids], **sampling},
                       sim_code_version=calibration.SIM_CODE_VERSION, items=items, skipped=skipped)
