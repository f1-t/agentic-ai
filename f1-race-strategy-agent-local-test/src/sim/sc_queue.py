"""The safety-car queue (docs/eval.md "The safety-car queue").

Under a safety car the field closes up behind it: whatever the gaps were when it came out, at the
end of its last lap the cars on the lead lap run nose to tail (the calibration races: median
consecutive gap at the line 0.49 s, the whole lead-lap field within 7 s; under a VSC the gaps are
held, median 4.8 s before and after). A replay that keeps each car's own time through the
period keeps gaps the race erased: on the 18 calibration races the neutralised laps carry most
of the judge's relative gap error (15.5 s of it; 5.2 s with them recorded), and the recorded SC
lap times alone make the order worse, because they close the real gap to the car ahead, not the
simulated one. What a safety car does is to GAPS and ORDER, so the queue is applied to the
field's cumulative times, not to any car's lap times:

  for each recorded SC period (laps S..E, a lap counted when most of the field's rows carry status
  4), in race order:
    order   the cars' simulated race time at the end of lap S - 1 (when it came out) plus the pit
            loss of every stop made in S..E (a stop under the safety car costs places in the queue,
            not open-air seconds); at S = 1 the simulated time at the end of lap E
    leader  the first car in that order; its time at the end of lap E is its order time plus the
            period's recorded length (the real leader's time from the end of lap S - 1 to the end
            of lap E; from the start when S = 1), when given, else its own simulated time
    queue   each following car's time at the end of lap E = the car ahead's + min(its gap in the
            order, spacing_s), so a car already closer than the spacing keeps its gap
    lapped  lapped cars rejoin the lead lap's queue (they are waved past to unlap: the real
            field at the end of an SC is within 7 s, median, every car included), except a car
            that finished laps down (`lapped`, the recorded classification: it stayed a lap down),
            which keeps its time
    after   every later lap of the car moves by the same shift; racing resumes from the queue

VSC periods change nothing here (the gaps are held). Cars that are not in the field at lap E (a
retirement) are left as they are.

The numbers: spacing_s 0.5, the median real gap at the end of the SC periods of the calibration
races (2019-2024, measured once; set a priori for any race scored, not tuned on an evaluation
result). Which laps were neutralised is the recorded race's (hindsight, the judge's ledger
"events"); a simulator that draws its periods applies the same queue to its drawn ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SC = "4"


def sc_periods(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """[(first, last lap)] of the recorded safety-car periods: laps where most rows carry status 4."""
    share = frame.groupby("lap")["track_status"].agg(lambda s: float(np.mean([SC in str(x) for x in s])))
    laps = sorted(int(L) for L in share.index[share > 0.5])
    out = []
    for L in laps:
        if out and L == out[-1][1] + 1:
            out[-1] = (out[-1][0], L)
        else:
            out.append((L, L))
    return out


def durations(frame: pd.DataFrame, periods) -> dict:
    """{(S, E): seconds} the recorded length of each period: the race leader at lap E's time from
    the end of its lap S - 1 (from the race start when S = 1) to the end of its lap E."""
    out = {}
    for S, E in periods:
        at = frame[frame["lap"] == E].dropna(subset=["end_s"])
        if not len(at):
            continue
        lead = at.loc[at["end_s"].idxmin()]
        g = frame[frame["driver"] == lead["driver"]].set_index("lap")
        t0 = float(frame["start_s"].min()) if S <= 1 else float(g["end_s"].get(S - 1, np.nan))
        if np.isfinite(t0):
            out[(S, E)] = float(lead["end_s"]) - t0
    return out


def apply(cum: dict, pit: dict, periods, spacing_s: float, lap_ref_s: float = None, lengths=None,
          lapped=()) -> dict:
    """{driver: shifted cumulative times}. `cum[d]` is a pd.Series of driver d's simulated race time
    at the end of each lap (index lap), `pit[d]` a pd.Series of the pit loss charged on each lap
    (0 where none), `lengths` durations(). Returns new Series, the input unchanged."""
    out = {d: s.astype(float).copy() for d, s in cum.items()}
    for S, E in periods:
        at_end = [d for d, s in out.items() if E in s.index]
        if len(at_end) < 2:
            continue
        end = pd.Series({d: float(out[d][E]) for d in at_end})
        if S > 1:
            ok = [d for d in at_end if (S - 1) in out[d].index]
            if len(ok) < 2:
                continue
            key = pd.Series({d: float(out[d][S - 1]) + float(pit[d].loc[S:E].sum()) for d in ok})
        else:
            key = end.copy()
        key = key.sort_values(kind="stable")
        order = list(key.index)
        lead = order[0]
        length = (lengths or {}).get((S, E))
        if length is None:
            new = {lead: float(end[lead])}
        else:
            new = {lead: (float(key[lead]) if S > 1 else 0.0) + length}
        prev = lead
        for d in order[1:]:
            if d in lapped:   # stayed a lap down: not in the lead lap's queue
                continue
            new[d] = new[prev] + min(float(key[d] - key[prev]), spacing_s)
            prev = d
        for d, t in new.items():
            s = out[d]
            s.loc[s.index >= E] += t - float(end[d])
    return out


def totals(cum: dict) -> pd.Series:
    """Each car's race time at its last lap."""
    return pd.Series({d: float(s.iloc[-1]) for d, s in cum.items() if len(s)})


# --- the single-car queue: one car's plans against the other cars' race ---------------------------
def runs(laps) -> list[tuple[int, int]]:
    """Contiguous runs of lap numbers as (first, last)."""
    out = []
    for L in sorted(int(x) for x in laps):
        if out and L == out[-1][1] + 1:
            out[-1] = (out[-1][0], L)
        else:
            out.append((L, L))
    return out


class FieldQueue:
    """The other cars of one snapshot's race for the single-car queue (the engine under
    sc_queue.mode "single"). `rows` are the rows the simulator may read: the whole race for the
    judge, the rows visible at the snapshot for a windowed fit (`horizon`: the last lap they show
    for the snapshot's car; the judge passes the race length). For a period that starts after the
    horizon the others are projected with their gaps to the car at the horizon held (they run at
    the car's own simulated pace, its stops apart).

    At a period S..E (S >= 2) each plan's car is placed in the queue by its key (race time at the
    end of lap S - 1 plus its pit loss charged in S..E) among the others' keys (their time at the
    end of lap S - 1 plus, per stop they made in S..E, the pit loss times the SC factor); its time
    at the end of lap E becomes the first key plus the period's length plus spacing_s per lead-lap
    car ahead of it. The length is the recorded one when `recorded` and the race had a safety car on
    every lap of S..E (the judge: the race leader at lap E's time from the end of its lap S - 1 to
    the end of lap E), else
    the car's own time from the end of lap S - 1 through its simulated SC laps. The car's key carries its pace (own_pace_s per simulated lap before the period) when the
    simulator adds none (Simulator.pace None, the eval's judge): the queue order is a fact about the
    car's real speed, which a field-mean car would get wrong. Lapped cars rejoin the queue, except
    the cars `lapped` names (the judge: cars that finished laps down); the car itself is left as it
    is when it finishes laps down (`own_lapped`), and so is a period with S = 1 (the start decides
    that order)."""

    def __init__(self, rows: pd.DataFrame, driver: str, horizon: int, spacing_s: float, lap_ref_s: float,
                 recorded: bool = False, own_pace_s: float = 0.0, lapped=(), own_lapped: bool = False):
        self.spacing_s, self.lap_ref_s, self.recorded = float(spacing_s), float(lap_ref_s), recorded
        self.own_pace_s = float(own_pace_s) if np.isfinite(own_pace_s) else 0.0
        self.own_lapped = bool(own_lapped)
        self.horizon = int(horizon)
        lap1 = rows[rows["lap"] == 1]["start_s"].dropna()
        self.start = float(lap1.min()) if len(lap1) else float(rows["start_s"].min())
        own = rows[rows["driver"] == driver]
        own1 = own.loc[own["lap"] == 1, "start_s"].dropna()
        self.own_offset = float(own1.iloc[0]) - self.start if len(own1) else 0.0
        self.own_t = (own.set_index("lap")["end_s"].astype(float) - self.start)
        self.sc_laps = {L for p in sc_periods(rows) for L in range(p[0], p[1] + 1)} if len(rows) else set()
        self.all_t = rows.pivot_table(index="lap", columns="driver", values="end_s", aggfunc="first") - self.start
        others = rows[(rows["driver"] != driver) & ~rows["driver"].isin(list(lapped))]
        self.t = others.pivot_table(index="lap", columns="driver", values="end_s", aggfunc="first") - self.start
        self.pit = others.assign(pit_in=others["pit_in"].astype(float)).pivot_table(
            index="lap", columns="driver", values="pit_in", aggfunc="first").fillna(0.0)

    def length(self, S, E) -> float | None:
        """The recorded time the race leader at lap E took from the end of its lap S - 1 to the end of
        lap E; None where the rows do not show it."""
        if E not in self.all_t.index or (S - 1) not in self.all_t.index:
            return None
        at_e = self.all_t.loc[E].dropna()
        if not len(at_e):
            return None
        lead = at_e.idxmin()
        t0 = self.all_t.loc[S - 1].get(lead, np.nan)
        return float(at_e[lead] - t0) if np.isfinite(t0) else None

    def at(self, lap: int) -> pd.Series:
        """The others' race time at the end of `lap`: recorded where the rows show it, else from the
        car's last shown lap plus lap_ref_s per lap to go (a car behind on the road that had not
        finished `lap` when a windowed fit's rows were cut); cars with no row up to `lap` left out."""
        t = self.t.loc[self.t.index <= lap]
        if not len(t):
            return pd.Series(dtype=float)
        last = t.apply(lambda c: c.last_valid_index())
        val = t.ffill().iloc[-1]
        ok = last.notna()
        return (val[ok] + (lap - last[ok].astype(float)) * self.lap_ref_s).astype(float)

    def shift(self, S, E, lap_s, pit_lap, first_lap, elapsed_s, pit_loss_s, sc_factor) -> np.ndarray:
        """Per plan, the time to add at the end of lap E (0 where the queue does not apply).
        lap_s, pit_lap: (n_plans, n_laps) lap times and pit loss charged per lap from first_lap on;
        elapsed_s: the car's time at the end of lap first_lap - 1 on its own clock."""
        n_plans = lap_s.shape[0]
        zero = np.zeros(n_plans)
        k = first_lap - 1
        if S < 2 or E < first_lap or self.own_lapped:
            return zero
        i0, iE = max(S - first_lap, 0), E - first_lap
        base = elapsed_s + self.own_offset
        run = np.cumsum(lap_s + pit_lap, axis=1) + base
        if S - 1 > k:   # the car's own pace on its simulated laps before the period (own_pace_s)
            t_before = run[:, S - 1 - first_lap] + self.own_pace_s * (S - 1 - k)
        elif S - 1 == k:
            t_before = np.full(n_plans, base)
        elif (S - 1) in self.own_t.index and np.isfinite(self.own_t[S - 1]):
            t_before = np.full(n_plans, float(self.own_t[S - 1]))
        else:
            return zero
        key_d = t_before + pit_lap[:, i0:iE + 1].sum(axis=1)
        if S - 1 <= self.horizon:
            t_o = self.at(S - 1)
            stops = self.pit.reindex(index=range(S, min(E, self.horizon) + 1), columns=t_o.index).fillna(0.0).sum(axis=0)
            key_o = np.broadcast_to((t_o + stops * pit_loss_s * sc_factor).to_numpy(dtype=float)[None, :],
                                    (n_plans, len(t_o)))
        else:   # projected from the horizon: the gaps to the car held, the car's stops apart
            own_h = base if self.horizon == k else float(self.own_t.get(self.horizon, np.nan))
            if not np.isfinite(own_h):
                return zero
            gaps = self.at(self.horizon).to_numpy(dtype=float) - own_h
            # the car's time at S - 1 without its stops
            drive = np.cumsum(lap_s, axis=1)[:, S - 1 - first_lap] + base + self.own_pace_s * (S - 1 - k)
            key_o = drive[:, None] + gaps[None, :]
        if key_o.shape[1] == 0:
            return zero
        first = np.minimum(key_o.min(axis=1), key_d)
        ahead = (key_o < key_d[:, None]).sum(axis=1)
        recorded = self.recorded and all(L in self.sc_laps for L in range(S, E + 1))
        length = self.length(S, E) if recorded else None
        if length is None:   # the car's own neutralised time from the end of lap S - 1
            length = run[:, iE] - t_before - pit_lap[:, i0:iE + 1].sum(axis=1)
        t_new = first + length + self.spacing_s * ahead
        # where own_pace_s stands in for a pace term the car's clock carries none: the shift lands
        # it at its place in the queue and its later laps run as before
        return t_new - run[:, iE]


def own_pace(model, g: pd.DataFrame) -> float:
    """The car's median lap time less the model's on its green non-pit laps after lap 1 (the rows
    given: the whole race for the judge, the visible ones for a windowed fit); 0 with fewer than 5."""
    c = g[(g["lap"] > 1) & (g["track_status"].astype(str) == "1") & ~g["pit_in"].astype(bool)
          & ~g["pit_out"].astype(bool) & g["lap_time_s"].notna() & g["tyre_age"].notna()
          & g["compound"].isin(list(model.compounds))]
    if len(c) < 5:
        return 0.0
    pred = np.empty(len(c))
    for comp, idx in c.groupby("compound").indices.items():
        sub = c.iloc[idx]
        pred[idx] = model.lap_times(comp, sub["tyre_age"].to_numpy(dtype=float), sub["lap"].to_numpy(dtype=float))
    return float(np.median(c["lap_time_s"].to_numpy(dtype=float) - pred))
