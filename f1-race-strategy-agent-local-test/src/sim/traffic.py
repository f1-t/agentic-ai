"""Traffic: a track-position penalty against the recorded field, replayed as ghost cars.

A traffic model maps to the engine through two calls, mirroring track.py:

    model.prepare(snapshot, laps, frame, driver, ref_lap_s=None) -> state
                                                              once per batch call, plan- and
                                                              sample-independent; None when
                                                              traffic cannot be placed
    model.apply(state, lap_s, pit_s_lap, sc, vsc, scenario) -> (lap_s, extra)
                                                              per sample, after the lap time is
                                                              complete (overlay, lap 1 extra,
                                                              hazard, pace) and before pit loss

`lap_s` is (n_plans, n_laps) over the remaining laps, `pit_s_lap` the pit loss booked on
each lap (in-laps only), `sc` and `vsc` the sample's SC and VSC laps (bool, (n_laps,)).
apply returns lap_s with the traffic increments added and extra = (traffic_s, passes,
bunching_s, sc_field_delta): the first three (n_plans,), traffic_s the green-lap
increments, bunching_s the SC-lap ones (below); sc_field_delta (n_plans, n_laps) is the
field's SC lap less the engine's on each bunched SC lap, NaN elsewhere. Traffic
draws nothing, so every seed keeps its draws. model.support_warnings(snapshot, state)
gives the strings added to every plan's support warnings.

"none" (the default) prepares None, and apply is never called.

"ghost" (source "recorded") replays every other driver's recorded laps as ghosts that do
not react to us. WORLD-MODEL ONLY: the ghosts' remaining laps are the future, so this
belongs to ground truth, calibration, replay and the eval's scorer, never to the belief
side (reads_future True).

Projected ghosts (source "projected", the belief's traffic; reads_future False). prepare
cuts the frame to lap <= k = snapshot.laps_completed before it reads anything, then
project_field projects every other car with a lap-k row: it laps at `model` (the engine's
degradation fit) for its compound and age plus its median residual over its last pace_laps
green laps (its median lap where the model lacks the compound); a car that has not stopped
stops once when its set reaches the field's typical life so far (typical_life: the median
tyre age at the stops made up to k, per compound, else over every compound, else half the
race), at the latest min_stint_laps before the flag; a car that has stopped stops again
only when age + remaining laps > stretch x that life. A projected stop adds the race's pit
loss to the out-lap and fits next_compound (the compound most used at the stops so far). A
car whose lap k is an in-lap comes out on lap k + 1. Variant 0 is that central projection;
n_variants - 1 more shift each stop by round(N(0, stop_sd_laps)) and each pace by N(0, its
standard error, at least pace_sd_floor_s), from default_rng([PROJECTION_SEED, k]); the k-th
apply call of a batch takes variant k mod n_variants, so the samples carry the projection's
spread. None at k = 0 (nobody has a lap yet).

Neutral alignment (align_neutral; always on for the projected source). The recorded
ghosts ran the recorded SC/VSC laps; a sample whose neutral laps differ (the world draws no
events, so a recorded SC is a green lap for us) would meet a field slowed or sped up by
laps it does not run. On such a lap (never a bunched SC lap) every ghost's clock from that
lap on is shifted by one common delta, so the field's gaps are kept: record neutral and
sample green, delta = a green field lap (green_ref: the field's median lap on the nearest
earlier green lap, else later, else ref_lap_s) less the field's median lap on it; sample
neutral and record green, delta = our median lap on it less the field's median lap. Where
the sample's neutral laps are the record's nothing moves.

own_pace: our clock among the ghosts runs at f + own_offset_s on green laps (pace_offset:
the driver's median residual over the model, free-air green laps when there are
OWN_PACE_MIN_LAPS of them; recorded: the whole race, projected: up to k). It decides whom we
catch and our positions; the returned lap times are f plus the traffic increments, so a
pace-corrected replay (calibration.midrace's err_offset_s) adds the driver term once.

pass_mode "earned": of the cars `passing` would pass, a pass completes only when arrive +
(passes tried) x pass_cost_s <= G - pass_gap_s; the others are attacked, which counts as
dirty air (when dirty_air_s > 0), and end = max(clamp, arrive + extra): no swap, so no lap
ends earlier than our own arrival and no time is conjured. A car only a little faster sits
in dirty air for more laps than a much faster one. dirty_air_auto sets dirty_air_s from the
data (dirty_air_estimate; recorded: the whole race, projected: up to k).

Positions (state.positions after each apply, BatchResult.final_pos / rejoin_pos /
pit_window_cars): 1 + the ghosts whose (shifted) G crossed the line before us at the end of
the plan's first out-lap and of the last lap, and the ghosts whose G at the first in-lap's
line is within that stop's pit loss behind ours (they come by while we are in the pit lane).

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

SC bunching (sc_bunching, off by default: on Spain 2021 it failed the 0013 decision rule,
see docs/traffic.md). Under the SC the field closes up into a queue
about sc_interval_s apart, so on an SC lap we take our place in it rather than keep our
pre-SC time gap. The queue is the ghosts in it on this lap: both ends recorded, not on
their in-lap (they leave the queue for the pit lane; out-laps rejoin and stay), and within
half a reference lap of our clock, |G_prev - T| < 0.5 * ref_lap_s (the same-lap-count
guard: a lapped car's lap-i end, about a lap earlier in time, is not the car ahead).
The car's lap time is the field's recorded SC lap on this lap,
f_field = median over the queue's ghosts not on their out-lap of G - G_prev, so our clock
and the ghosts' are comparable on a lap where the recorded queue ran far from the engine's
flat SC lap; the engine's f is the fallback when no ghost qualifies. The provisional end is
e = T + f_field + pit_s_lap (a pit under the SC is booked before the placement, so we join
the queue where the reduced pit loss puts us). In the queue, A is directly ahead of e
(largest G <= e) and B directly behind (smallest G > e):

    end = G_A + min(sc_interval_s, (G_B - G_A) / 2)    A and B exist
        = G_A + sc_interval_s                          A only
        = e                                            no ghost ahead (we lead the queue)

lap_s = end - T - pit_s_lap, so the pit loss stays out of the lap time. The placement
part (end - e) is negative when we close up to the queue and positive when we were closer
to A than the interval. Order is kept (we stay between A and B, never past B), and no car
is passed on an SC lap; repeating it on every SC lap converges like a real queue. The
increments over the engine's lap (f_field - f plus end - e) go to bunching_s, not
traffic_s. With bunching on, the engine's SC lap factor therefore only matters with
traffic off. A VSC keeps the gaps (no bunching). The SC laps are the
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
    sc_bunching False      0013's decision rule: on only if it improved post-SC placement
                           without worsening the gate and pairs; on Spain it did not
    sc_interval_s 1.2      about ten car lengths under the SC, 1 to 1.5 s at SC speed

Known limits, not solved here:
(1) SC laps keep their time gaps by default (time-based SC): a car 20 s ahead of a ghost
    before the SC is still 20 s ahead after it while the recorded ghosts are bunched.
    sc_bunching is the queue rule, available but off. With it on, the restart is not
    modelled (a car keeps its slot), and under drawn events the recorded field's lap on a
    drawn SC lap is usually green, so f_field is not an SC lap there.
(2) Lapped cars: ghosts are compared on the same lap number, so a car a lap down is
    "behind" in lap index even when ahead on track; blue flags make this second order.
(3) Ghosts do not react: no undercut defence, no rival strategist.
(4) Our clock drifts from the real one by the model error (0.1 to 0.3 s/lap), so positions
    far from the snapshot are noisy; the rejoin a few laps on is the trustworthy use.
(5) Recorded ghosts are valid only under the recorded events. Under a drawn SC the field ran
    green, so any field-time rule prices that SC at nothing. With an event model other than
    "none", or a fixed scenario other than the record over the remaining laps, the engine
    adds "traffic_recorded_with_drawn_events" to every plan's support warnings (positions
    and traffic approximate). The optimizer with rate events keeps traffic off; the ghost
    model is for replay, calibration and hindsight ground truth under the real events.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
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

    def prepare(self, snapshot, laps, frame, driver, ref_lap_s=None, model=None):
        return None

    def apply(self, state, lap_s, pit_s_lap, sc, vsc, scenario):
        raise RuntimeError("apply is never called when prepare returns None")

    def support_warnings(self, snapshot, state):
        return []


class GhostParams(BaseModel):
    """The "ghost" model's ComponentSpec params. See the module docstring for the sources."""

    model_config = ConfigDict(extra="forbid")

    # recorded: the other cars' recorded laps (world model only, it reads the future);
    # projected: the other cars projected forward from the laps up to the snapshot (belief)
    source: Literal["recorded", "projected"] = "recorded"
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
    sc_bunching: bool = False   # SC queue (0013), available but off by default
    sc_interval_s: float = Field(default=1.2, ge=0)   # queue interval under the SC
    # neutral-lap alignment: on a lap neutral (SC/VSC) in the sample but not in the ghosts'
    # record, or the reverse, the ghosts' common clock is shifted so the field gaps are kept
    # (module docstring, "Neutral alignment"); a no-op where the sample's neutral laps are
    # the record's, so replays under the real events are unchanged, and on bunched SC laps,
    # which take the field's own SC lap. Off by default (the earlier behaviour); the eval's
    # world config turns it on
    align_neutral: bool = False
    # projected source only (module docstring, "Projected ghosts")
    n_variants: int = Field(default=8, ge=1)          # projection variants, cycled over samples
    pace_laps: int = Field(default=10, ge=1)          # a ghost's pace: its green laps in this window
    stop_sd_laps: float = Field(default=3.0, ge=0)    # variant spread of a projected stop lap
    pace_sd_floor_s: float = Field(default=0.15, ge=0)  # variant spread of a ghost's pace, at least
    stretch: float = Field(default=1.3, ge=1)         # a stopped ghost runs to the flag within stretch x typical life
    # our own pace on the track: our clock among the ghosts runs at the model's lap plus the
    # driver's median residual over the model (free-air green laps; recorded: the whole race,
    # projected: the laps up to the snapshot). It moves who we catch and our positions only:
    # the lap times returned stay the model's plus the traffic increments (module docstring)
    own_pace: bool = False
    # "swap" (the earlier rule): a pass ends pass_gap_s ahead of the car passed, which can
    # grant time the car did not earn (conjured). "earned": a pass completes only on a lap
    # whose own arrival, pass costs included, is pass_gap_s ahead of the car; until then the
    # car attacks from behind in its dirty air (dirty_air_s), so a small pace advantage costs
    # more laps behind than a large one and no lap gains time from a pass
    pass_mode: Literal["swap", "earned"] = "swap"
    # dirty_air_s from the data: the mean residual of green laps within dirty_air_range_s of
    # the car ahead less that of free-air green laps, over the model (recorded: the whole race;
    # projected: the laps up to the snapshot), clipped to [0, DIRTY_AIR_MAX_S]; dirty_air_s
    # when either side has fewer than DIRTY_AIR_MIN_LAPS laps
    dirty_air_auto: bool = False


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


def place(frame, driver, lap, t):
    """(position, gap): our position at the end of `lap` if we cross the line at session time
    t, 1 + the ghosts (every other driver in `frame`) that finished `lap` before t, and t less
    the end of lap `lap` of the car directly ahead (NaN when none is). A ghost without a lap
    `lap` (retired, or laps down) is not ahead. calibration.rejoin and placement rank our
    clock with it, the eval's position regret our finishing time (SPEC 0014 D)."""
    G = ghost_table(frame, driver, [lap]).G[:, 0]
    ahead = G[np.isfinite(G) & (G < t)]
    return 1 + len(ahead), (float(t - ahead.max()) if len(ahead) else float("nan"))


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


def _neutral_status(status):
    """SC (4) or VSC (6, 7) in a track status string, per row."""
    s = pd.Series(status).astype(str)
    return (s.str.contains("4") | s.str.contains("[67]")).to_numpy()


def recorded_neutral(frame, driver, laps) -> np.ndarray:
    """(n_laps,) bool: lap L of the record is neutral (SC or VSC), from `driver`'s own row when
    it has one (the same rule as calibration._scenario, so a replay under the real events
    finds no mismatch), else from the field's majority on that lap."""
    laps = np.asarray(laps, dtype=int)
    if "track_status" not in frame or not len(laps):
        return np.zeros(len(laps), dtype=bool)
    neutral = pd.Series(_neutral_status(frame["track_status"]), index=frame.index)
    field_major = neutral.groupby(frame["lap"]).mean() > 0.5
    mine = frame["driver"] == driver
    own = neutral[mine].groupby(frame.loc[mine, "lap"]).first()
    return np.array([bool(own[L]) if L in own.index else bool(field_major.get(L, False)) for L in laps])


def _green_ref(g_med, rec_neutral, fallback):
    """A green field lap for every lap: g_med on the record's green laps; on a neutral lap the
    nearest earlier green lap's g_med, else the nearest later one, else `fallback`."""
    n = len(g_med)
    ok = ~rec_neutral & np.isfinite(g_med)
    out = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        if ok[i]:
            last = g_med[i]
        out[i] = last
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if ok[i]:
            nxt = g_med[i]
        if not np.isfinite(out[i]):
            out[i] = nxt
    return np.where(np.isfinite(out), out, float(fallback))


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
    half_lap_s: float = np.inf    # SC queue's same-lap-count guard, 0.5 * ref_lap_s
    # neutral alignment (n_laps,): the record is neutral on the lap; the field's median ghost
    # lap (eligible, not on an out-lap; NaN without one); a green field lap for that lap
    rec_neutral: np.ndarray = None
    g_med: np.ndarray = None
    green_ref: np.ndarray = None
    # projected source: one state per projection variant (variants[0] is this state, the
    # central projection); apply cycles through them, the k-th call taking variant k mod n
    variants: list = None
    calls: int = 0
    own_offset_s: float = 0.0   # our pace over the model on the track (own_pace), 0 when off
    # the last apply call's positions per plan: final_pos, rejoin_pos, pit_window_cars
    positions: dict = None
    dirty_air_s: float = 0.0    # the params' dirty_air_s, or dirty_air_auto's estimate


def _fill(state: GhostState, table: GhostTable, p: GhostParams) -> GhostState:
    """The per-lap rows of `table` on `state`, and its field median lap."""
    state.table = table
    state.G_prev = np.ascontiguousarray(table.G_prev.T)
    state.G = np.ascontiguousarray(table.G.T)
    state.G_follow = state.G + p.follow_gap_s
    state.G_swap = state.G - p.pass_gap_s
    state.g = np.ascontiguousarray(table.g.T)
    state.eligible = np.ascontiguousarray(
        (np.isfinite(table.G_prev) & np.isfinite(table.G) & ~table.in_lap).T)
    state.out_lap = state.eligible & table.out_lap.T
    timed = state.eligible & ~state.out_lap
    state.g_med = np.full(len(table.laps), np.nan)
    has = timed.any(axis=1)
    if has.any():
        state.g_med[has] = np.nanmedian(np.where(timed[has], state.g[has], np.nan), axis=1)
    return state


# --- projected ghosts (the belief's traffic) ---------------------------------------------
PROJECTION_SEED = 12_002   # the projection variants' rng: default_rng([PROJECTION_SEED, laps_completed])
DRY_STOP_RULE = ("SOFT", "MEDIUM", "HARD")   # a dry race: a car must use two of these


def _green_rows(df):
    """Rows a pace can be read from, decided from the row itself (not the frame's clean flag,
    whose filter saw the whole race): after lap 1, green, no pit in or out, a lap time."""
    ts = df["track_status"].astype(str).to_numpy() if "track_status" in df else np.full(len(df), "1")
    lt = df["lap_time_s"].to_numpy(dtype=float)
    keep = ((df["lap"].to_numpy(dtype=float) > 1) & (ts == "1") & ~df["pit_in"].to_numpy(dtype=bool)
            & ~df["pit_out"].to_numpy(dtype=bool) & np.isfinite(lt))
    return df[keep]


def _model_times(model, compound, ages, lapnums):
    """model.lap_times, or None when `model` is None or does not model `compound`."""
    if model is None or compound not in getattr(model, "compounds", ()):
        return None
    return np.asarray(model.lap_times(compound, np.asarray(ages, dtype=float), np.asarray(lapnums, dtype=float)),
                      dtype=float)


OWN_PACE_MIN_LAPS = 5   # free-air laps needed for own_pace from them, else every green lap
DIRTY_AIR_MIN_LAPS = 20   # laps on each side for dirty_air_auto
DIRTY_AIR_MAX_S = 1.5


def _residuals(rows, model):
    """Lap time less the model's, per row of `rows` (NaN where the model lacks the compound)."""
    out = np.full(len(rows), np.nan)
    pos = np.arange(len(rows))
    for c, idx in rows.groupby("compound").indices.items():
        sub = rows.iloc[idx]
        pred = _model_times(model, str(c), pd.to_numeric(sub["tyre_age"], errors="coerce"), sub["lap"])
        if pred is not None:
            out[pos[idx]] = sub["lap_time_s"].to_numpy(dtype=float) - pred
    return out


def dirty_air_estimate(rows, model, range_s, fallback) -> float:
    """dirty_air_auto's value from `rows` (every driver's laps the estimate may read): the mean
    residual over `model` of green laps whose gap to the car ahead at the lap start is under
    `range_s`, less that of free-air green laps, clipped to [0, DIRTY_AIR_MAX_S]; `fallback`
    with fewer than DIRTY_AIR_MIN_LAPS laps on either side or without the gap columns."""
    if model is None or "gap_ahead_s" not in rows or "free_air" not in rows:
        return float(fallback)
    green = _green_rows(rows)
    r = _residuals(green, model)
    gap = green["gap_ahead_s"].to_numpy(dtype=float)
    ok = np.isfinite(r)
    pack = ok & np.isfinite(gap) & (gap < range_s)
    free = ok & green["free_air"].eq(True).to_numpy()
    if pack.sum() < DIRTY_AIR_MIN_LAPS or free.sum() < DIRTY_AIR_MIN_LAPS:
        return float(fallback)
    return float(np.clip(r[pack].mean() - r[free].mean(), 0.0, DIRTY_AIR_MAX_S))


def pace_offset(rows, model, prefer_free=True, min_laps=OWN_PACE_MIN_LAPS) -> float:
    """The median residual of one driver's green rows (_green_rows) over `model`, on its
    free-air rows when `prefer_free` and it has min_laps of them; 0 without a residual."""
    green = _green_rows(rows)
    if prefer_free and "free_air" in green:
        free = green[green["free_air"].eq(True)]
        if len(free) >= min_laps:
            green = free
    resid = []
    for c, sub in green.groupby("compound"):
        pred = _model_times(model, str(c), pd.to_numeric(sub["tyre_age"], errors="coerce"), sub["lap"])
        if pred is not None:
            resid.append(sub["lap_time_s"].to_numpy(dtype=float) - pred)
    resid = np.concatenate(resid) if resid else np.empty(0)
    resid = resid[np.isfinite(resid)]
    return float(np.median(resid)) if len(resid) else 0.0


def typical_life(past) -> tuple[dict, float | None]:
    """({compound: median tyre age at a stop}, the median over every compound) over the stops
    made in `past` (pit_in rows), ({}, None) without any."""
    stops = past[past["pit_in"].astype(bool)]
    age = pd.to_numeric(stops["tyre_age"], errors="coerce")
    ok = age.notna()
    if not ok.any():
        return {}, None
    by = {str(c): float(a.median()) for c, a in age[ok].groupby(stops.loc[ok, "compound"])}
    return by, float(age[ok].median())


def next_compound(past, current, model):
    """The compound a projected stop fits: the most used on out-laps so far other than
    `current` (ties alphabetical), else a modelled dry compound other than `current`, else
    `current`."""
    outs = past.loc[past["pit_out"].astype(bool), "compound"].astype(str)
    counts = outs[outs != current].value_counts()
    if len(counts):
        return sorted(counts[counts == counts.max()].index)[0]
    others = sorted(c for c in getattr(model, "compounds", ()) if c != current and c in DRY_STOP_RULE)
    return others[0] if others else current


@dataclass
class GhostProjection:
    """One other car's projection at the snapshot, before a variant's jitter."""
    driver: str
    start_s: float          # its crossing at the end of lap k
    compound: str
    age: float              # tyre age after lap k
    offset_s: float         # its pace over the model, the median residual of its recent green laps
    offset_sd_s: float      # that median's standard error (NaN below two laps)
    const_s: float          # its median recent green lap, the pace on a compound the model lacks
    stop_lap: int | None    # projected in-lap (> k), None for no further stop
    pitted_now: bool        # its lap k is an in-lap: the out-lap is k + 1
    out_extra_s: float      # time its out-lap adds over its pace
    next_compound: str


def project_field(snapshot, frame, driver, model=None, ref_lap_s=None, params: GhostParams | None = None):
    """GhostProjection per other car with a lap-k row and an end_s on it, from the rows of
    `frame` with lap <= k = snapshot.laps_completed only (nothing later is read). The stop rule
    (module docstring, "Projected ghosts"): a car that has not stopped stops when its set
    reaches the field's typical life so far (the median tyre age at the stops made so far, per
    compound), at the latest min_stint_laps before the flag; a car that has stopped stops again
    only when its set would pass `stretch` times that life before the flag."""
    p = params or GhostParams(source="projected")
    k, total = snapshot.laps_completed, snapshot.race.total_laps
    frame = frame[frame["lap"] <= k]
    min_stint = int(getattr(snapshot.race, "min_stint_laps", 3) or 3)
    pit = float(snapshot.race.pit_loss_s or 0.0)
    past = frame[frame["driver"] != driver]
    now = past[(past["lap"] == k) & np.isfinite(past["end_s"].to_numpy(dtype=float))]
    life_by, life_all = typical_life(frame)
    green_all = _green_rows(past[past["lap"] > k - p.pace_laps])
    field_lap = (float(green_all["lap_time_s"].median()) if len(green_all)
                 else float(ref_lap_s) if ref_lap_s is not None and np.isfinite(ref_lap_s) else np.nan)
    out = []
    for d in sorted(now["driver"].unique()):
        rows = past[past["driver"] == d].sort_values("lap")
        last = rows.iloc[-1]
        comp = str(last["compound"])
        age = float(pd.to_numeric(pd.Series([last.get("tyre_age")]), errors="coerce").iloc[0])
        if not np.isfinite(age):
            outs = rows.loc[rows["pit_out"].astype(bool), "lap"]
            age = float(k - (outs.max() - 1 if len(outs) else 0))
        green = _green_rows(rows)
        recent = green[green["lap"] > k - p.pace_laps]
        use = recent if len(recent) >= 3 else green
        const = float(use["lap_time_s"].median()) if len(use) else field_lap
        resid = []
        for c, sub in use.groupby("compound"):
            ages = pd.to_numeric(sub["tyre_age"], errors="coerce").fillna(age)
            pred = _model_times(model, str(c), ages, sub["lap"])
            if pred is not None:
                resid.append(sub["lap_time_s"].to_numpy(dtype=float) - pred)
        resid = np.concatenate(resid) if resid else np.empty(0)
        resid = resid[np.isfinite(resid)]
        offset = float(np.median(resid)) if len(resid) else 0.0
        offset_sd = float(np.std(resid) / np.sqrt(len(resid))) if len(resid) >= 2 else np.nan
        n_stops = int(rows["pit_in"].astype(bool).sum())
        pitted_now = bool(last["pit_in"])
        life = life_by.get(comp, life_all)
        if life is None:
            life = float(np.ceil(total / 2))
        stop = None
        if not pitted_now:
            must = n_stops == 0 and comp in DRY_STOP_RULE
            if must or age + (total - k) > p.stretch * life:
                stop = int(min(k + max(1, round(life - age)), total - min_stint))
                if stop <= k:
                    stop = k + 1 if total - (k + 1) >= 1 else None
        extra = pit
        if pitted_now:
            pred = _model_times(model, comp, [age], [k])
            expected = (pred[0] + offset) if pred is not None else const
            lt = float(last["lap_time_s"])
            extra = max(0.3 * pit, pit - max(0.0, lt - expected)) if np.isfinite(lt) and np.isfinite(expected) else pit
        out.append(GhostProjection(d, float(last["end_s"]), comp, age, offset, offset_sd, const, stop,
                                   pitted_now, extra, next_compound(past, comp, model)))
    return out


def projected_table(projections, laps, total, model, pit, variant=None, min_stint=3) -> GhostTable:
    """The GhostTable of `projections` over the remaining `laps`. `variant` is [(stop shift in
    laps, pace shift in s)] per car for a jittered variant, None for the central projection.
    A projected stop adds the pit loss to the out-lap; its in-lap is never caught."""
    laps = np.asarray(laps, dtype=int)
    ghosts = [g.driver for g in projections]
    n, n_l = len(ghosts), len(laps)
    G = np.full((n, n_l), np.nan)
    G_prev = np.full((n, n_l), np.nan)
    in_lap = np.zeros((n, n_l), dtype=bool)
    out_lap = np.zeros((n, n_l), dtype=bool)
    if not n_l:
        return GhostTable(ghosts, laps, G, G_prev, G - G_prev, in_lap, out_lap)
    k = int(laps[0]) - 1
    for j, gp in enumerate(projections):
        stop, offset = gp.stop_lap, gp.offset_s
        if variant is not None:
            ds, do = variant[j]
            offset = offset + do
            if stop is not None:
                stop = int(np.clip(stop + ds, k + 1, max(k + 1, total - min_stint)))
        times = np.empty(n_l)
        if gp.pitted_now:
            segs = [(0, n_l, gp.next_compound, 1.0)]
        elif stop is None or stop >= laps[-1]:
            segs = [(0, n_l, gp.compound, gp.age + 1)]
        else:
            s = int(stop - laps[0]) + 1   # index of the out-lap
            segs = [(0, s, gp.compound, gp.age + 1), (s, n_l, gp.next_compound, 1.0)]
        for a, b, comp, age0 in segs:
            pred = _model_times(model, comp, age0 + np.arange(b - a), laps[a:b])
            times[a:b] = pred + offset if pred is not None else gp.const_s
        if gp.pitted_now:
            out_lap[j, 0] = True
            times[0] += gp.out_extra_s
        elif stop is not None and stop < laps[-1]:
            i = int(stop - laps[0])
            in_lap[j, i] = True
            out_lap[j, i + 1] = True
            times[i + 1] += pit
        start = gp.start_s + np.concatenate([[0.0], np.cumsum(times)])
        G_prev[j], G[j] = start[:-1], start[1:]
    return GhostTable(ghosts, laps, G, G_prev, G - G_prev, in_lap, out_lap)


@TRAFFIC.register("ghost")
class GhostTraffic:
    def __init__(self, **params):
        self.params = GhostParams(**params)
        # the recorded source replays the other cars' future laps: world model only
        self.reads_future = self.params.source == "recorded"

    def prepare(self, snapshot, laps, frame, driver, ref_lap_s=None, model=None):
        """`ref_lap_s` is the engine's reference lap for the SC queue's half-lap guard; None
        (direct calls) takes the median recorded lap of the eligible ghosts not on an out-lap.
        `model` (the engine's degradation fit) prices the projected source's laps; the
        recorded source does not read it."""
        p = self.params
        if p.source == "projected":
            return self._prepare_projected(snapshot, laps, frame, driver, ref_lap_s, model)
        t0 = anchor(frame, snapshot)
        if t0 is None:
            return None
        table = ghost_table(frame, driver, laps)
        state = _fill(GhostState(table=table, anchor=t0, sc_history=list(snapshot.sc_laps)), table, p)
        state.rec_neutral = recorded_neutral(frame, driver, laps)
        if p.own_pace:
            state.own_offset_s = pace_offset(frame[frame["driver"] == driver], model)
        state.dirty_air_s = (dirty_air_estimate(frame, model, p.dirty_air_range_s, p.dirty_air_s)
                             if p.dirty_air_auto else p.dirty_air_s)
        return self._finish(state, snapshot, ref_lap_s)

    def _prepare_projected(self, snapshot, laps, frame, driver, ref_lap_s, model):
        """The other cars projected from the laps up to the snapshot. The frame is cut to
        lap <= laps_completed before anything is read, so no later lap reaches the state.
        None at the start (no lap yet) or without elapsed_s."""
        p, k = self.params, snapshot.laps_completed
        if snapshot.elapsed_s is None or k < 1:
            return None
        past = frame[frame["lap"] <= k]
        t0 = anchor(past, snapshot)
        proj = project_field(snapshot, past, driver, model, ref_lap_s, p)
        total, pit = snapshot.race.total_laps, float(snapshot.race.pit_loss_s or 0.0)
        min_stint = int(getattr(snapshot.race, "min_stint_laps", 3) or 3)
        table = projected_table(proj, laps, total, model, pit, min_stint=min_stint)
        state = _fill(GhostState(table=table, anchor=t0, sc_history=list(snapshot.sc_laps)), table, p)
        state.rec_neutral = np.zeros(len(table.laps), dtype=bool)
        if p.own_pace:
            state.own_offset_s = pace_offset(past[past["driver"] == driver], model)
        state.dirty_air_s = (dirty_air_estimate(past, model, p.dirty_air_range_s, p.dirty_air_s)
                             if p.dirty_air_auto else p.dirty_air_s)
        state = self._finish(state, snapshot, ref_lap_s)
        state.variants = [state]
        rng = np.random.default_rng([PROJECTION_SEED, k])
        for _ in range(p.n_variants - 1):
            jitter = [(int(round(rng.normal(0.0, p.stop_sd_laps))),
                       float(rng.normal(0.0, p.pace_sd_floor_s if np.isnan(g.offset_sd_s)
                                        else max(p.pace_sd_floor_s, g.offset_sd_s))))
                      for g in proj]
            vt = projected_table(proj, laps, total, model, pit, variant=jitter, min_stint=min_stint)
            v = _fill(GhostState(table=vt, anchor=t0, sc_history=state.sc_history), vt, p)
            v.rec_neutral, v.half_lap_s = state.rec_neutral, state.half_lap_s
            v.green_ref = _green_ref(v.g_med, v.rec_neutral, 2 * state.half_lap_s)
            state.variants.append(v)
        return state

    def _finish(self, state, snapshot, ref_lap_s):
        """half_lap_s, green_ref and the anchor check."""
        table = state.table
        if ref_lap_s is None:
            laps_g = state.g[state.eligible & ~state.out_lap]
            ref_lap_s = float(np.median(laps_g)) if laps_g.size else np.inf
        state.half_lap_s = 0.5 * float(ref_lap_s)
        state.green_ref = _green_ref(state.g_med, state.rec_neutral, ref_lap_s)
        if (snapshot.car_ahead is not None and snapshot.gap_ahead_s is not None
                and snapshot.car_ahead in table.ghosts and len(table.laps)):
            implied = state.anchor - table.G_prev[table.ghosts.index(snapshot.car_ahead), 0]
            diff = implied - snapshot.gap_ahead_s
            if abs(diff) > ANCHOR_TOL_S:
                state.warnings.append(f"traffic_anchor_mismatch:{diff:.1f}")
        return state

    def support_warnings(self, snapshot, state):
        if state is None:
            return ["traffic_unavailable:no_elapsed_s" if snapshot.elapsed_s is None
                    else "traffic_unavailable:no_laps"]
        return list(state.warnings)

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

    def _bunch(self, G, e, queue):
        """(n_plans,) end of an SC lap from the provisional end `e`: behind the queue's ghost
        directly ahead of e at the queue interval, never past the one behind (module
        docstring). `queue` (n_plans, n_ghosts) is the ghosts in the queue."""
        a = np.where(queue & (G <= e[:, None]), G, -np.inf).max(axis=1, initial=-np.inf)
        b = np.where(queue & (G > e[:, None]), G, np.inf).min(axis=1, initial=np.inf)
        has_a, has_b = np.isfinite(a), np.isfinite(b)
        a = np.where(has_a, a, 0.0)   # finite stand-ins, so no inf arithmetic warns
        interval = self.params.sc_interval_s
        interval = np.where(has_b, np.minimum(interval, 0.5 * (np.where(has_b, b, 0.0) - a)), interval)
        return np.where(has_a, a + interval, e)

    def apply(self, state, lap_s, pit_s_lap, sc, vsc, scenario):
        p = self.params
        st = state.variants[state.calls % len(state.variants)] if state.variants else state
        state.calls += 1
        lap_s = np.array(lap_s, dtype=float)
        pit_s_lap = np.asarray(pit_s_lap, dtype=float)
        n_plans, n_laps = lap_s.shape
        sc, vsc = np.asarray(sc, dtype=bool), np.asarray(vsc, dtype=bool)
        neutral = sc | vsc
        bunch = sc if p.sc_bunching else np.zeros(n_laps, dtype=bool)
        drs = self.drs_on(state, neutral, scenario)
        # a projected field has no neutral laps of its own, so it always takes the sample's
        align = p.align_neutral or p.source == "projected"
        mismatch = ((neutral != st.rec_neutral) & ~bunch) if align else np.zeros(n_laps, dtype=bool)
        T = np.full(n_plans, state.anchor)
        traffic_s = np.zeros(n_plans)
        bunching_s = np.zeros(n_plans)
        passes = np.zeros(n_plans, dtype=int)
        field_delta = np.full((n_plans, n_laps), np.nan)
        # positions: at the end of each plan's first in-lap, of the out-lap after it, and at the flag
        stop = pit_s_lap > 0
        first_in = np.where(stop.any(axis=1), stop.argmax(axis=1), -2)
        rejoin_pos = np.full(n_plans, np.nan)
        window_cars = np.full(n_plans, np.nan)
        final_pos = np.full(n_plans, np.nan)
        dirty = state.dirty_air_s
        dirty_on = dirty > 0
        earned = p.pass_mode == "earned"
        S = 0.0   # the ghosts' common clock shift so far (neutral alignment)
        own = state.own_offset_s
        for i in range(n_laps):
            f = lap_s[:, i]
            # our clock on the track: the model's lap plus our own pace on green laps (own_pace)
            ftrk = f + own if own and not neutral[i] else f
            arrive = T + ftrk
            Gp, G, Gf, Gs, g = st.G_prev[i], st.G[i], st.G_follow[i], st.G_swap[i], st.g[i]
            delta = 0.0
            if mismatch[i] and np.isfinite(st.g_med[i]):
                target = float(np.median(f)) if neutral[i] else float(st.green_ref[i])
                delta = target - float(st.g_med[i])
            if S or delta:
                Gp = Gp + S
                G, Gf, Gs, g = G + (S + delta), Gf + (S + delta), Gs + (S + delta), g + delta
                S += delta
            if bunch[i]:
                queue = st.eligible[i] & (np.abs(Gp - T[:, None]) < state.half_lap_s)
                timed = queue & ~st.out_lap[i]
                has = timed.any(axis=1)
                f_sc = f.copy()
                if has.any():   # rows without a timed ghost keep the engine's f (no all-NaN median)
                    f_sc[has] = np.nanmedian(np.where(timed[has], g, np.nan), axis=1)
                    field_delta[has, i] = f_sc[has] - f[has]
                e = T + f_sc + pit_s_lap[:, i]
                end = self._bunch(G, e, queue)
                # the leader keeps e: its lap is f_sc bit for bit (end - e is exactly 0)
                new = f_sc + (end - e)
                bunching_s += new - f
                lap_s[:, i] = new
                cross, T = end - pit_s_lap[:, i], end
            elif neutral[i]:
                cross = arrive
                T = arrive + pit_s_lap[:, i]
            else:
                d = T[:, None] - Gp
                behind = (d > 0) & st.eligible[i]
                catch = behind & (arrive[:, None] < Gf)
                if not catch.any() and not dirty_on:
                    cross = arrive
                    T = arrive + pit_s_lap[:, i]
                else:
                    if drs[i]:
                        thresh = np.where(d <= p.drs_range_s + _GAP_EPS, p.pass_delta_drs_s, p.pass_delta_s)
                    else:
                        thresh = p.pass_delta_s
                    fast = g - ftrk[:, None] >= thresh
                    out = st.out_lap[i]
                    # an out-lap ghost is never passed: held when we arrive after its line, else nothing
                    held = catch & np.where(out, arrive[:, None] > G, ~fast)
                    clamp = self._clamp(held, Gf)
                    passing = catch & fast & ~out & (Gs >= clamp[:, None])
                    over = passing.sum(axis=1) > p.max_passes_per_lap
                    if over.any():
                        # the closest (largest G_prev) are passed, the rest hold us
                        key = np.where(passing, Gp, -np.inf)
                        rank = np.argsort(np.argsort(-key, axis=1, kind="stable"), axis=1)
                        keep = passing & (rank < p.max_passes_per_lap)
                        held = held | (passing & ~keep)
                        clamp = self._clamp(held, Gf)
                        passing = keep & (Gs >= clamp[:, None])
                    if earned:
                        # a pass completes when the arrival with every pass cost is pass_gap_s
                        # ahead of the car; the others are attacked from behind, in dirty air
                        tried = passing.sum(axis=1)
                        done = passing & ((arrive + tried * p.pass_cost_s)[:, None] <= Gs)
                        attacking = passing & ~done
                        passing = done
                    n_pass = passing.sum(axis=1)
                    extra = n_pass * p.pass_cost_s
                    if dirty_on:
                        near = behind & (d <= p.dirty_air_range_s + _GAP_EPS) & ~passing
                        if earned:
                            near = near | attacking
                        extra = extra + dirty * near.any(axis=1)
                    if earned:   # no swap: a lap never ends earlier than the car's own arrival
                        inc = np.maximum(clamp - arrive, extra)
                    else:
                        swap = np.where(passing, Gs, np.inf).min(axis=1, initial=np.inf)
                        # end = max(clamp, min(arrive + extra, swap)), kept as an increment so a lap
                        # without traffic keeps its lap time bit for bit; negative when the swap binds
                        inc = np.maximum(clamp - arrive, np.minimum(extra, swap - arrive))
                    lap_s[:, i] = f + inc
                    traffic_s += inc
                    passes += n_pass
                    cross = arrive + inc
                    T = cross + pit_s_lap[:, i]
            # positions at the line at the end of lap i: 1 + the ghosts that crossed it earlier
            w = first_in == i
            if w.any():   # the cars that come by while we are in the pit lane: within the pit loss behind
                c = cross[w, None]
                window_cars[w] = ((G[None, :] > c) & (G[None, :] <= c + pit_s_lap[w, i, None])).sum(axis=1)
            r = first_in + 1 == i
            if r.any() or i == n_laps - 1:
                ahead = 1 + (G[None, :] < cross[:, None]).sum(axis=1)
                rejoin_pos[r] = ahead[r]
                if i == n_laps - 1:
                    final_pos = ahead.astype(float)
        # this call's positions, read by the engine after the call (the extra tuple keeps its shape)
        state.positions = {"final_pos": final_pos, "rejoin_pos": rejoin_pos, "pit_window_cars": window_cars}
        return lap_s, (traffic_s, passes, bunching_s, field_delta)


