"""Traffic: a track-position penalty against the recorded field, replayed as ghost cars.

A traffic model maps to the engine through two calls, mirroring track.py:

    model.prepare(snapshot, laps, frame, driver) -> state     once per batch call, plan- and
                                                              sample-independent; None when
                                                              traffic cannot be placed
    model.apply(state, lap_s, pit_s_lap, neutral, scenario) -> (lap_s, extra)
                                                              per sample, after the lap time is
                                                              complete (overlay, lap 1 extra,
                                                              hazard, pace) and before pit loss

`lap_s` is (n_plans, n_laps) over the remaining laps, `pit_s_lap` the pit loss booked on
each lap (in-laps only), `neutral` the sample's SC or VSC laps. apply returns lap_s with
the traffic increments added and extra = (traffic_s, passes), both (n_plans,). Traffic
draws nothing, so every seed keeps its draws. model.support_warnings(snapshot, state)
gives the strings added to every plan's support warnings.

"none" (the default) prepares None, and apply is never called.

"ghost" (source "recorded") replays every other driver's recorded laps as ghosts that do
not react to us. WORLD-MODEL ONLY: the ghosts' remaining laps are the future, so this
belongs to ground truth, calibration and replay, never to the agent's belief side. The
agent's tools run with traffic off until a projected-ghost source exists (0012b; source
"projected" is reserved for it).

Ghost table (ghost_table). Common clock = session time. Over the remaining lap numbers L:
G[j, L] is the end_s of ghost j's lap L (NaN where absent), G_prev[j, L] the end_s of its
lap L - 1 (the lap-k row for the first remaining lap; NaN without one, lap 0 included),
g = G - G_prev its recorded lap time, in_lap[j, L] its pit_in on L. A ghost's in-lap never
clamps (its time has the pit entry in it); its out-lap does, since being held behind a car
on cold tyres is the undercut victim's reality. Our anchor is t0 + snapshot.elapsed_s,
with t0 the race start: the snapshot driver's lap-1 start_s, else the field minimum.

Per sample, sequential over the remaining laps, vectorised over plans and ghosts, with T
our clock at the lap start and f our lap time as the engine produced it:

    neutral lap (SC or VSC): no traffic, end = T + f
    else  d       = T - G_prev                          gap at the lap start, per ghost
          behind  = d > 0, ghost eligible (both ends recorded, not its in-lap)
          catch   = behind & (T + f < G + follow_gap_s)  we would reach the car this lap
          thresh  = pass_delta_drs_s when DRS is on and d <= drs_range_s, else pass_delta_s
          passing = catch & (g - f >= thresh), held = catch & ~passing
          end     = max(T + f + min(#passing, max_passes_per_lap) * pass_cost_s + dirty,
                        max over held of G + follow_gap_s)
    lap_s = end - T; then T = end + pit loss, so the out-lap starts from the rejoin position.

DRS is off on race laps 1 to drs_off_first_laps, on SC and VSC laps, and for
drs_off_after_sc_laps after the last lap of an SC period (the snapshot's SC history
counts). dirty is dirty_air_s while any ghost ahead is within dirty_air_range_s and we are
not passing (off by default).

Constants, a priori and frozen; sources are placeholders, not facts the code depends on,
and all of them will be selected by leave-one-race-out across races later:
    follow_gap_s 1.0       drivers and commentary put a stable following gap near 1 s
    drs_range_s 1.0        DRS detection uses a 1.0 s gap
    pass_delta_s 0.8       Barcelona is hard to pass on, hence 0.8 rather than the 0.5
                           often quoted for easy tracks
    pass_delta_drs_s 0.3
    pass_cost_s 0.4        the order of a lap lost per ten overtakes
    dirty_air_range_s 2.0  drivers and commentary put the dirty-air window near 2 s

Known limits, not solved here:
(1) SC bunching: the SC model is time-based, so a plan 20 s ahead of a ghost before the SC
    is still 20 s ahead after it while the recorded ghosts are bunched; positions on the
    laps after an SC are approximate. A position-based SC is the 0013 candidate.
(2) Lapped cars: ghosts are compared on the same lap number, so a car a lap down is
    "behind" in lap index even when ahead on track; blue flags make this second order.
(3) Ghosts do not react: no undercut defence, no rival strategist.
(4) Our clock drifts from the real one by the model error (0.1 to 0.3 s/lap), so positions
    far from the snapshot are noisy; the rejoin a few laps on is the trustworthy use.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .registry import Registry

TRAFFIC = Registry("traffic model")
ANCHOR_TOL_S = 1.0    # snapshot gap_ahead_s against the anchor-implied gap
# a held car sits at exactly follow_gap_s, which equals drs_range_s by default; the gap is
# a difference of session times, so compare with a tolerance or float rounding decides DRS
_GAP_EPS = 1e-9


@TRAFFIC.register("none")
class NoTraffic:
    def __init__(self, **_):
        pass

    def prepare(self, snapshot, laps, frame, driver):
        return None

    def apply(self, state, lap_s, pit_s_lap, neutral, scenario):
        raise RuntimeError("apply is never called when prepare returns None")

    def support_warnings(self, snapshot, state):
        return []


class GhostParams(BaseModel):
    """The "ghost" model's ComponentSpec params. See the module docstring for the sources."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["recorded", "projected"] = "recorded"   # projected: reserved for 0012b
    follow_gap_s: float = Field(default=1.0, ge=0)
    drs_range_s: float = Field(default=1.0, ge=0)
    pass_delta_s: float = 0.8
    pass_delta_drs_s: float = 0.3
    pass_cost_s: float = Field(default=0.4, ge=0)
    drs_off_first_laps: int = Field(default=2, ge=0)
    drs_off_after_sc_laps: int = Field(default=2, ge=0)
    dirty_air_s: float = Field(default=0.0, ge=0)
    dirty_air_range_s: float = Field(default=2.0, ge=0)
    max_passes_per_lap: int = Field(default=3, ge=0)


@dataclass
class GhostTable:
    """Ghost arrays over the remaining laps, each (n_ghosts, n_laps). See the module docstring."""

    ghosts: list
    laps: np.ndarray
    G: np.ndarray
    G_prev: np.ndarray
    g: np.ndarray
    in_lap: np.ndarray


def ghost_table(frame, driver, laps) -> GhostTable:
    """Every driver in `frame` except `driver`, over the lap numbers `laps`."""
    laps = np.asarray(laps, dtype=int)
    ghosts = sorted(d for d in frame["driver"].unique() if d != driver)
    n_lap = int(max(laps.max(), frame["lap"].max())) + 1 if len(laps) else 1
    end = np.full((len(ghosts), n_lap), np.nan)
    pit = np.zeros((len(ghosts), n_lap), dtype=bool)
    rows = frame[frame["driver"].isin(ghosts)]
    j = np.searchsorted(ghosts, rows["driver"].to_numpy())
    lap = rows["lap"].to_numpy(dtype=int)
    end[j, lap] = rows["end_s"].to_numpy(dtype=float)
    pit[j, lap] = rows["pit_in"].to_numpy(dtype=bool)
    G, G_prev = end[:, laps], end[:, laps - 1]   # lap 0 has no row, so NaN
    return GhostTable(ghosts=ghosts, laps=laps, G=G, G_prev=G_prev, g=G - G_prev, in_lap=pit[:, laps])


def race_start(frame, driver):
    """t0 on the session clock: the driver's lap-1 start_s, else the field minimum of lap-1 start_s."""
    first = frame[frame["lap"] == 1]
    own = first.loc[first["driver"] == driver, "start_s"]
    if len(own) and np.isfinite(own.iloc[0]):
        return float(own.iloc[0])
    return float(first["start_s"].min())


def anchor(frame, snapshot):
    """Our clock at the start of lap laps_completed + 1, None without snapshot.elapsed_s."""
    if snapshot.elapsed_s is None:
        return None
    return race_start(frame, snapshot.driver) + float(snapshot.elapsed_s)


def _runs(laps):
    """Last lap of each run of consecutive lap numbers."""
    laps = sorted(set(int(x) for x in laps))
    return [x for i, x in enumerate(laps) if i == len(laps) - 1 or laps[i + 1] != x + 1]


@dataclass
class GhostState:
    table: GhostTable
    anchor: float
    sc_history: list
    warnings: list = field(default_factory=list)
    # per lap rows (n_laps, n_ghosts), contiguous for the lap loop
    G_prev: np.ndarray = None
    G_follow: np.ndarray = None   # G + follow_gap_s
    g: np.ndarray = None
    eligible: np.ndarray = None


@TRAFFIC.register("ghost")
class GhostTraffic:
    def __init__(self, **params):
        self.params = GhostParams(**params)
        if self.params.source != "recorded":
            raise NotImplementedError(f"ghost source {self.params.source!r} is reserved for 0012b")

    def prepare(self, snapshot, laps, frame, driver):
        t0 = anchor(frame, snapshot)
        if t0 is None:
            return None
        table = ghost_table(frame, driver, laps)
        state = GhostState(table=table, anchor=t0, sc_history=list(snapshot.sc_laps))
        state.G_prev = np.ascontiguousarray(table.G_prev.T)
        state.G_follow = np.ascontiguousarray(table.G.T + self.params.follow_gap_s)
        state.g = np.ascontiguousarray(table.g.T)
        state.eligible = np.ascontiguousarray(
            (np.isfinite(table.G_prev) & np.isfinite(table.G) & ~table.in_lap).T)
        if (snapshot.car_ahead is not None and snapshot.gap_ahead_s is not None
                and snapshot.car_ahead in table.ghosts and len(table.laps)):
            implied = t0 - table.G_prev[table.ghosts.index(snapshot.car_ahead), 0]
            diff = implied - snapshot.gap_ahead_s
            if abs(diff) > ANCHOR_TOL_S:
                state.warnings.append(f"traffic_anchor_mismatch:{diff:.1f}")
        return state

    def support_warnings(self, snapshot, state):
        return ["traffic_unavailable:no_elapsed_s"] if state is None else list(state.warnings)

    def drs_on(self, state, neutral, scenario):
        """(n_laps,) bool: DRS enabled on each remaining lap for this sample."""
        p, laps = self.params, state.table.laps
        on = (laps > p.drs_off_first_laps) & ~np.asarray(neutral, dtype=bool)
        for last in _runs([*state.sc_history, *scenario.sc_laps]):
            on &= ~((laps > last) & (laps <= last + p.drs_off_after_sc_laps))
        return on

    def apply(self, state, lap_s, pit_s_lap, neutral, scenario):
        p = self.params
        lap_s = np.array(lap_s, dtype=float)
        n_plans, n_laps = lap_s.shape
        drs = self.drs_on(state, neutral, scenario)
        T = np.full(n_plans, state.anchor)
        traffic_s = np.zeros(n_plans)
        passes = np.zeros(n_plans, dtype=int)
        dirty_on = p.dirty_air_s > 0
        for i in range(n_laps):
            f = lap_s[:, i]
            arrive = T + f
            if neutral[i]:
                T = arrive + pit_s_lap[:, i]
                continue
            d = T[:, None] - state.G_prev[i]
            behind = (d > 0) & state.eligible[i]
            catch = behind & (arrive[:, None] < state.G_follow[i])
            if not catch.any() and not dirty_on:
                T = arrive + pit_s_lap[:, i]
                continue
            if drs[i]:
                thresh = np.where(d <= p.drs_range_s + _GAP_EPS, p.pass_delta_drs_s, p.pass_delta_s)
            else:
                thresh = p.pass_delta_s
            passing = catch & (state.g[i] - f[:, None] >= thresh)
            held = catch & ~passing
            n_pass = np.minimum(passing.sum(axis=1), p.max_passes_per_lap)
            extra = n_pass * p.pass_cost_s
            if dirty_on:
                near = behind & (d <= p.dirty_air_range_s + _GAP_EPS) & ~passing
                extra = extra + p.dirty_air_s * near.any(axis=1)
            clamp = np.where(held, state.G_follow[i], -np.inf).max(axis=1, initial=-np.inf)
            # end = max(arrive + extra, clamp), kept as an increment so a lap without traffic
            # keeps its lap time bit for bit
            inc = np.maximum(extra, clamp - arrive)
            lap_s[:, i] = f + inc
            traffic_s += inc
            passes += n_pass
            T = arrive + inc + pit_s_lap[:, i]
        return lap_s, (traffic_s, passes)
