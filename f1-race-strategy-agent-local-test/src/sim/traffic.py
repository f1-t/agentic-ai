"""Traffic: a track-position penalty against the recorded field, replayed as ghost cars.

A traffic model maps to the engine through two calls, mirroring track.py:

    model.prepare(snapshot, laps, frame, driver) -> state     once per batch call, plan- and
                                                              sample-independent; None when
                                                              traffic cannot be placed
    model.apply(state, lap_s, pit_s_lap, sc, vsc, scenario) -> (lap_s, extra)
                                                              per sample, after the lap time is
                                                              complete (overlay, lap 1 extra,
                                                              hazard, pace) and before pit loss

`lap_s` is (n_plans, n_laps) over the remaining laps, `pit_s_lap` the pit loss booked on
each lap (in-laps only), `sc` and `vsc` the sample's SC and VSC laps (bool, (n_laps,)).
apply returns lap_s with the traffic increments added and extra = (traffic_s, passes,
bunching_s), each (n_plans,): traffic_s the green-lap increments, bunching_s the SC-lap
ones (below). Traffic
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
g = G - G_prev its recorded lap time, in_lap[j, L] its pit_in on L, out_lap[j, L] its
pit_out on L. A ghost's in-lap never clamps (its time has the pit entry in it). A ghost on
its out-lap is never passed on track (it rejoined from the pit lane): if we arrive before
its G we are simply ahead (it rejoined behind us; no cost, no swap), if we arrive within
follow_gap_s after its G we are held behind the car that rejoined in front of us (the
undercut victim's reality), later than that nothing. Our anchor is t0 +
snapshot.elapsed_s, with t0 the race start: the snapshot driver's lap-1 start_s, else the
field minimum.

Per sample, sequential over the remaining laps, vectorised over plans and ghosts, with T
our clock at the lap start, f our lap time as the engine produced it and arrive = T + f:

    VSC lap (and SC lap with sc_bunching False): no traffic, end = arrive
    SC lap with sc_bunching: the queue, below
    else  d       = T - G_prev                          gap at the lap start, per ghost
          behind  = d > 0, ghost eligible (both ends recorded, not its in-lap)
          catch   = behind & (arrive < G + follow_gap_s)  we would reach the car this lap
          thresh  = pass_delta_drs_s when DRS is on and d <= drs_range_s, else pass_delta_s
          fast    = g - f >= thresh
          held    = catch & ~fast (not on its out-lap)  |  catch & out-lap & (arrive > G)
          clamp   = max over held of (G + follow_gap_s), -inf if none
          passing = catch & fast & passable & (G - pass_gap_s >= clamp)
                    passable = not on its out-lap; never past the car that holds us
          cap:      more than max_passes_per_lap passing: the closest ones (largest G_prev)
                    are passed, the rest count as held (clamp recomputed, the kept passes
                    rechecked against it)
          free    = arrive + #passing * pass_cost_s + dirty
          swap    = min over passing of (G - pass_gap_s), +inf if none
          end     = max(clamp, min(free, swap))
    lap_s = end - T; then T = end + pit loss, so the out-lap starts from the rejoin position.

A pass is a position swap: it always ends at least pass_gap_s ahead of every car passed,
so the next lap's d is negative for them and a pass is never charged twice. When free >
swap the swap grants time the car did not earn, at most follow_gap_s + pass_gap_s +
pass_cost_s - (g - f) on that lap, so a lap's increment can be negative ("conjured"
time). That is the price of ghosts that do not react: in a reactive world the passed car
would lose that time instead. calibrate traffic reports the conjured total.

SC bunching (sc_bunching, on by default). Under the SC the field closes up into a queue
about sc_interval_s apart, so on an SC lap we take our place in it rather than keep our
pre-SC time gap. With f the SC lap time as the engine gives it (the leader's, the same for
every car), the provisional end is e = T + f + pit_s_lap (a pit under the SC is booked
before the placement, so we join the queue where the reduced pit loss puts us). Among the
ghosts with a recorded G on this lap, A is directly ahead of e (largest G <= e) and B
directly behind (smallest G > e):

    end = G_A + min(sc_interval_s, (G_B - G_A) / 2)    A and B exist
        = G_A + sc_interval_s                          A only
        = e                                            no ghost ahead (we lead the queue)

lap_s = end - T - pit_s_lap, so the pit loss stays out of the lap time. The increment is
negative when we close up to the queue and positive when we were closer to A than the
interval. Order is kept (we stay between A and B, never past B), and no car is passed on
an SC lap; repeating it on every SC lap converges like a real queue. The increments go to
bunching_s, not traffic_s. A VSC keeps the gaps (no bunching). The SC laps are the
engine's, scenario.sc_laps, which for drawn events already starts inside the snapshot's
SC.

DRS is off on race laps 1 to drs_off_first_laps, on SC and VSC laps, and for
drs_off_after_sc_laps after the last lap of an SC period (the snapshot's SC history
counts). A held car sits at follow_gap_s = drs_range_s, which is in DRS range by design
(compared with a 1e-9 s tolerance). dirty is dirty_air_s while any ghost ahead is within
dirty_air_range_s and we are not passing (off by default).

Constants, a priori and frozen; sources are placeholders, not facts the code depends on,
and all of them will be selected by leave-one-race-out across races later:
    follow_gap_s 1.0       drivers and commentary put a stable following gap near 1 s
    drs_range_s 1.0        DRS detection uses a 1.0 s gap
    pass_delta_s 0.8       Barcelona is hard to pass on, hence 0.8 rather than the 0.5
                           often quoted for easy tracks
    pass_delta_drs_s 0.3
    pass_cost_s 0.4        the order of a lap lost per ten overtakes
    pass_gap_s 0.2         a car length and a bit at the line; a pass never ends in a tie
    dirty_air_range_s 1.5  FREE_AIR_GAP_S (config): dirty_air_s is measured on clean_pack
                           laps, gaps under that, so it is applied over the same window
    sc_bunching True
    sc_interval_s 1.2      about ten car lengths under the SC, 1 to 1.5 s at SC speed

Known limits, not solved here:
(1) SC bunching is a queue rule (above), not a model of the restart: a car keeps its slot
    in the queue, nobody gains on the restart lap beyond the green-lap rule, and lapped
    cars are queued by lap index like everything else here.
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

from .config import FREE_AIR_GAP_S
from .data import line_crossings
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

    def apply(self, state, lap_s, pit_s_lap, sc, vsc, scenario):
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
    pass_gap_s: float = Field(default=0.2, ge=0)   # a pass ends at least this far ahead
    drs_off_first_laps: int = Field(default=2, ge=0)
    drs_off_after_sc_laps: int = Field(default=2, ge=0)
    dirty_air_s: float = Field(default=0.0, ge=0)
    dirty_air_range_s: float = Field(default=FREE_AIR_GAP_S, ge=0)
    max_passes_per_lap: int = Field(default=3, ge=0)
    sc_bunching: bool = True
    sc_interval_s: float = Field(default=1.2, ge=0)   # queue interval under the SC


@dataclass
class GhostTable:
    """Ghost arrays over the remaining laps, each (n_ghosts, n_laps). See the module docstring."""

    ghosts: list
    laps: np.ndarray
    G: np.ndarray
    G_prev: np.ndarray
    g: np.ndarray
    in_lap: np.ndarray
    out_lap: np.ndarray


def ghost_table(frame, driver, laps) -> GhostTable:
    """Every driver in `frame` except `driver`, over the lap numbers `laps`.

    The session times come from data.line_crossings, the end_s(L - 1) arithmetic shared
    with data.gap_columns: G_prev[j, L] is ghost j's crossing at the start of lap L, G its
    crossing at the start of lap L + 1. The anchor check in prepare and calibration.rejoin
    / placement read gaps only through this table.
    """
    laps = np.asarray(laps, dtype=int)
    ghosts = sorted(d for d in frame["driver"].unique() if d != driver)
    n_lap = int(max(laps.max(), frame["lap"].max())) + 2 if len(laps) else 2
    start = np.full((len(ghosts), n_lap), np.nan)   # start[j, L]: crossing at the start of lap L
    pit_in = np.zeros((len(ghosts), n_lap), dtype=bool)
    pit_out = np.zeros((len(ghosts), n_lap), dtype=bool)
    rows = frame[frame["driver"].isin(ghosts)]
    cross = line_crossings(rows)
    jc = np.searchsorted(ghosts, cross["driver"].to_numpy())
    start[jc, cross["lap"].to_numpy(dtype=int)] = cross["t"].to_numpy()
    j = np.searchsorted(ghosts, rows["driver"].to_numpy())
    lap = rows["lap"].to_numpy(dtype=int)
    pit_in[j, lap] = rows["pit_in"].to_numpy(dtype=bool)
    pit_out[j, lap] = rows["pit_out"].to_numpy(dtype=bool)
    G, G_prev = start[:, laps + 1], start[:, laps]   # no lap-0 row, so lap 1's G_prev is NaN
    return GhostTable(ghosts=ghosts, laps=laps, G=G, G_prev=G_prev, g=G - G_prev, in_lap=pit_in[:, laps],
                      out_lap=pit_out[:, laps])


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
    G: np.ndarray = None
    G_follow: np.ndarray = None   # G + follow_gap_s
    G_swap: np.ndarray = None     # G - pass_gap_s
    g: np.ndarray = None
    eligible: np.ndarray = None
    out_lap: np.ndarray = None    # eligible and on its out-lap: held or nothing, never passed


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
        state.G = np.ascontiguousarray(table.G.T)
        state.G_follow = state.G + self.params.follow_gap_s
        state.G_swap = state.G - self.params.pass_gap_s
        state.g = np.ascontiguousarray(table.g.T)
        state.eligible = np.ascontiguousarray(
            (np.isfinite(table.G_prev) & np.isfinite(table.G) & ~table.in_lap).T)
        state.out_lap = state.eligible & table.out_lap.T
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

    @staticmethod
    def _clamp(held, G_follow):
        """(n_plans,) max over held ghosts of G + follow_gap_s, -inf where none."""
        return np.where(held, G_follow, -np.inf).max(axis=1, initial=-np.inf)

    def _bunch(self, G, e):
        """(n_plans,) end of an SC lap from the provisional end `e`: behind the ghost directly
        ahead of e at the queue interval, never past the one behind (module docstring)."""
        G = G[np.isfinite(G)]
        a = np.where(G <= e[:, None], G, -np.inf).max(axis=1, initial=-np.inf)
        b = np.where(G > e[:, None], G, np.inf).min(axis=1, initial=np.inf)
        has_a, has_b = np.isfinite(a), np.isfinite(b)
        a = np.where(has_a, a, 0.0)   # finite stand-ins, so no inf arithmetic warns
        interval = self.params.sc_interval_s
        interval = np.where(has_b, np.minimum(interval, 0.5 * (np.where(has_b, b, 0.0) - a)), interval)
        return np.where(has_a, a + interval, e)

    def apply(self, state, lap_s, pit_s_lap, sc, vsc, scenario):
        p = self.params
        lap_s = np.array(lap_s, dtype=float)
        n_plans, n_laps = lap_s.shape
        sc, vsc = np.asarray(sc, dtype=bool), np.asarray(vsc, dtype=bool)
        neutral = sc | vsc
        bunch = sc if p.sc_bunching else np.zeros(n_laps, dtype=bool)
        drs = self.drs_on(state, neutral, scenario)
        T = np.full(n_plans, state.anchor)
        traffic_s = np.zeros(n_plans)
        bunching_s = np.zeros(n_plans)
        passes = np.zeros(n_plans, dtype=int)
        dirty_on = p.dirty_air_s > 0
        for i in range(n_laps):
            f = lap_s[:, i]
            arrive = T + f
            if bunch[i]:
                e = arrive + pit_s_lap[:, i]
                end = self._bunch(state.G[i], e)
                # the leader's lap keeps its time bit for bit: inc is exactly 0 where end is e
                inc = end - e
                lap_s[:, i] = f + inc
                bunching_s += inc
                T = end
                continue
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
            fast = state.g[i] - f[:, None] >= thresh
            out = state.out_lap[i]
            # an out-lap ghost is never passed: held when we arrive after its line, else nothing
            held = catch & np.where(out, arrive[:, None] > state.G[i], ~fast)
            clamp = self._clamp(held, state.G_follow[i])
            passing = catch & fast & ~out & (state.G_swap[i] >= clamp[:, None])
            over = passing.sum(axis=1) > p.max_passes_per_lap
            if over.any():
                # the closest (largest G_prev) are passed, the rest hold us
                key = np.where(passing, state.G_prev[i], -np.inf)
                rank = np.argsort(np.argsort(-key, axis=1, kind="stable"), axis=1)
                keep = passing & (rank < p.max_passes_per_lap)
                held = held | (passing & ~keep)
                clamp = self._clamp(held, state.G_follow[i])
                passing = keep & (state.G_swap[i] >= clamp[:, None])
            n_pass = passing.sum(axis=1)
            extra = n_pass * p.pass_cost_s
            if dirty_on:
                near = behind & (d <= p.dirty_air_range_s + _GAP_EPS) & ~passing
                extra = extra + p.dirty_air_s * near.any(axis=1)
            swap = np.where(passing, state.G_swap[i], np.inf).min(axis=1, initial=np.inf)
            # end = max(clamp, min(arrive + extra, swap)), kept as an increment so a lap
            # without traffic keeps its lap time bit for bit; negative when the swap binds
            inc = np.maximum(clamp - arrive, np.minimum(extra, swap - arrive))
            lap_s[:, i] = f + inc
            traffic_s += inc
            passes += n_pass
            T = arrive + inc + pit_s_lap[:, i]
        return lap_s, (traffic_s, passes, bunching_s)
