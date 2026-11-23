"""The driver's tyre sets at a snapshot (SimConfig.tyre_inventory), from the weekend's timing data.

With the switch off every snapshot carries sets_available=None and both simulators price each
later stint on a NEW set. In the real races a later SOFT stint starts on a used set about two
times in three. With it on, a snapshot gets the sets the driver still holds at that moment:

  allocation  the event's sets per compound: ALLOCATION_STANDARD (2 HARD, 3 MEDIUM, 8 SOFT) from
              2020 on, ALLOCATION_SPRINT (2, 4, 6) on a sprint weekend; 2019 allocations were
              chosen per driver and are not in the cache, so a 2019 race takes ALLOCATION_OPEN (13
              of each compound): unconstrained on new sets, but the used sets its sessions show are
              held at their ages
  minus       the sets opened in practice, sprint and qualifying, each kept at the laps it has run
              (tyre_sets.weekend_sets, which follows one physical set by its TyreLife)
  minus       the sets fitted in the race up to the snapshot, from the race laps completed by then
              (the driver's lap k end time; nothing later is read)

Sets handed back after practice are NOT modelled (returns = {}): the rule changed by season and
the teams pick which set goes back, so every opened set is kept as a used set. The estimate errs
toward allowing a set rather than forbidding one.

Every source is pre-race or race laps already completed at the snapshot: the inventory is as-of
(docs/eval.md "Information contract"), and the judge reads the same one.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import tyre_sets
from .race_ids import parse_race_id

ALLOCATION_STANDARD = {"HARD": 2, "MEDIUM": 3, "SOFT": 8}
ALLOCATION_SPRINT = {"HARD": 2, "MEDIUM": 4, "SOFT": 6}
ALLOCATION_OPEN = {"HARD": 13, "MEDIUM": 13, "SOFT": 13}   # 2019: the per-driver choice is not cached
FIRST_FIXED_SEASON = 2020
WEEKEND_SESSIONS = ("FP1", "FP2", "FP3", "SQ", "S", "Q")
Q2_START_RULE_LAST_SEASON = 2021


def _load(season, rnd, kind):
    from .data import load_session
    try:
        laps = load_session(season, rnd, kind).laps
    except Exception:   # not held, or not cached
        return None
    return laps if laps is not None and len(laps) else None


@lru_cache(maxsize=8)
def weekend(race_id: str):
    """(sessions, race laps, allocation) for a race id, or None when it cannot be built (a
    synthetic race, a race id that names no cached weekend, no race laps): such a race stays
    unconstrained. sessions are the pre-race FastF1 Laps tables keyed as tyre_sets wants them."""
    try:
        season, rnd, _ = parse_race_id(race_id)
    except (ValueError, TypeError):
        return None
    if int(season) > 2050:   # synthetic test races (2099_01_synthetic)
        return None
    race = _load(season, rnd, "R")
    if race is None:
        return None
    sessions = {k: v for k in WEEKEND_SESSIONS if (v := _load(season, rnd, k)) is not None}
    sprint = "S" in sessions or "SQ" in sessions
    if int(season) < FIRST_FIXED_SEASON:
        return sessions, race, dict(ALLOCATION_OPEN)
    return sessions, race, dict(ALLOCATION_SPRINT if sprint else ALLOCATION_STANDARD)


def visible_race_laps(race: pd.DataFrame, driver: str, k: int) -> pd.DataFrame:
    """Race laps completed by the end of `driver`'s lap k (session time), every car; k = 0 is the
    start: no lap is complete yet, but the first stints are known (the grid), so the cars' lap-1
    rows are kept for their compound and TyreLife only."""
    if k <= 0:
        return race[race["LapNumber"] == 1]
    own = race[(race["Driver"] == driver) & (race["LapNumber"] == k)]
    t = own["Time"].dt.total_seconds().to_numpy(dtype=float)
    if not len(t) or not np.isfinite(t[0]):
        return race[(race["Driver"] == driver) & (race["LapNumber"] <= k)]
    from .data import visible_rows   # the one cut by session time (end_s <= t), as every as-of input
    seen = visible_rows(race.assign(end_s=race["Time"].dt.total_seconds()), float(t[0]))
    return seen.drop(columns="end_s")


def sets_at(race_id: str, driver: str, k: int):
    """The driver's held sets at the end of his lap k ([{compound, laps_run, source, set_id}]),
    the fitted one left out; None when the race stays unconstrained."""
    w = weekend(race_id)
    if w is None:
        return None
    sessions, race, allocation = w
    season = parse_race_id(race_id)[0]
    laps = visible_race_laps(race, driver, k)
    if not (laps["Driver"] == driver).any():
        return None
    kw = dict(race_laps=laps, upto_lap=max(int(k), 1), q2_start_rule=int(season) <= Q2_START_RULE_LAST_SEASON)
    first = tyre_sets.weekend_report(sessions, driver, allocation, {}, **kw)
    extra = phantoms(first["notes"])
    if not extra:
        return first["sets"]
    # a used set that matched no held one was created outside the allocation's count: it may be a
    # set counted twice, so it does not use up one of the allocation's unseen (new) sets
    bumped = {c: n + extra.get(c, 0) for c, n in allocation.items()}
    return tyre_sets.weekend_sets(sessions, driver, bumped, {}, **kw)


def phantoms(notes) -> dict:
    """{compound: sets created because a used set matched no held one} from weekend_report notes."""
    out = {}
    for n in notes:
        if " matches no held set, created " in n:
            c = n.split(": ", 1)[1].split(" at TyreLife", 1)[0]
            out[c] = out.get(c, 0) + 1
    return out


def with_sets(snapshot, race_id: str | None = None):
    """`snapshot` with sets_available filled from sets_at (unchanged when the race stays
    unconstrained or the snapshot already carries an inventory)."""
    if snapshot.sets_available is not None:
        return snapshot
    rid = race_id or snapshot.race.race_id
    sets = sets_at(rid, snapshot.driver, int(snapshot.laps_completed))
    if sets is None:
        return snapshot
    from ..contracts import TyreSet
    known = {"SOFT", "MEDIUM", "HARD"}
    return snapshot.model_copy(update={"sets_available": [TyreSet(**s) for s in sets if s["compound"] in known]})


def on(*configs) -> bool:
    """Whether any of these SimConfigs (None skipped) turns the inventory on."""
    return any(getattr(c, "tyre_inventory", False) for c in configs if c is not None)
