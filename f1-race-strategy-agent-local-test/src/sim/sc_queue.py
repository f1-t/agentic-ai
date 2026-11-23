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

The near-lapped rule (ScQueue.held_behind_laps, off by default): a car more than held_behind_laps
(1.25) laps behind the leader when the safety car came out does not join the train. On the
calibration races' SC periods every car under one lap behind at the deployment closed up, and so
did most cars just past one lap (the wave-by lets them round in time); cars 1.3 to 1.6 laps
behind (2022 Bahrain: six cars, 24 to 38 s behind at the end; 2022 Brazil: one, 15 s) did not,
except in the longest period (2022 Canada, 7 laps). A held car keeps its gap in the order to the
train's last car ahead of it. Distances at the deployment come from the line crossings made
before it (deployment_times): a recorded crossing of lap S - 1 made under the safety car would
put a car behind cars it was ahead of, so the judge's and a windowed fit's others are keyed by
their green time at S - 1 from those crossings.

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


def deployment_times(rows: pd.DataFrame, S: int, lap_ref_s: float = None,
                     whole: bool = False) -> tuple[pd.Series, float] | None:
    """({driver: green race time at the end of lap S - 1}, lap_s) for the safety car whose first lap
    is S (the near-lapped rule, ScQueue.held_behind_laps), from the line crossings made before it:
    the deployment is after the last crossing with no status 4 that ends before the first crossing
    with one, and a car's time is its last crossing before then plus lap_s per lap to S - 1, lap_s
    the race leader's green lap (median of its last five green laps; lap_ref_s when it has none).
    The recorded crossing of lap S - 1 of a car far behind is made under the safety car; this is
    the time it would have had at green pace, the same footing as a simulated car's, so the order
    and distances are those at the deployment. Times in the rows' clock (end_s). None when the rows
    show no status 4 on laps S - 2..S + 1 (a windowed fit cut before the deployment: the caller
    falls back on its times at S - 1). Only cars still running at the deployment: a row of theirs
    ends after it, or (`whole` False: rows cut at a snapshot, where a car behind may not have
    crossed the line since) they are within a lap of the most laps completed."""
    if not len(rows) or "start_s" not in rows:
        return None
    near = rows[rows["lap"].between(S - 2, S + 1) & rows["end_s"].notna()]
    st = near["track_status"].astype(str)
    sc = near[st.str.contains(SC)]
    if not len(sc):
        return None
    first = sc["end_s"].idxmin()
    t_hi = float(sc.at[first, "end_s"])
    t_dep = float(sc.at[first, "start_s"]) if np.isfinite(sc.at[first, "start_s"]) else -np.inf
    pre = near[~st.str.contains(SC) & (near["end_s"] < t_hi)]
    if len(pre):
        t_dep = max(t_dep, float(pre["end_s"].max()))
    T = rows[rows["lap"] <= S + 1].pivot_table(index="lap", columns="driver", values="end_s", aggfunc="first")
    done = T.where(T <= t_dep).apply(lambda c: c.last_valid_index())
    done = done.dropna().astype(int)
    if not len(done):
        return None
    # cars still running at the deployment (a retired car's last crossing would project it forward
    # as if racing)
    last_end = rows.groupby("driver")["end_s"].max()
    on = last_end.reindex(done.index) > t_dep
    done = done[on if whole else on | (done >= done.max() - 1)]
    ld = int(done.max())
    at_ld = T.loc[ld, done.index[done == ld]].dropna()
    lead = at_ld.idxmin()
    g = rows[(rows["driver"] == lead) & rows["lap"].between(ld - 4, ld) & (rows["track_status"].astype(str) == "1")
             & ~rows["pit_in"].astype(bool) & ~rows["pit_out"].astype(bool)]["lap_time_s"].dropna()
    lap_s = float(g.median()) if len(g) else lap_ref_s
    if not lap_s or not np.isfinite(lap_s):
        return None
    return pd.Series({d: float(T.at[Lc, d]) + (S - 1 - Lc) * lap_s for d, Lc in done.items()}, dtype=float), lap_s


def deployment_behind(rows: pd.DataFrame, S: int, lap_ref_s: float = None) -> pd.Series | None:
    """{driver: laps behind the race leader at the deployment} (deployment_times: the time behind
    the first car over the leader's green lap); None where deployment_times is None."""
    got = deployment_times(rows, S, lap_ref_s)
    if got is None:
        return None
    t, lap_s = got
    return (t - t.min()) / lap_s


def apply(cum: dict, pit: dict, periods, spacing_s: float, lap_ref_s: float = None, lengths=None,
          lapped=(), held_laps: float = None, behind: dict = None) -> dict:
    """{driver: shifted cumulative times}. `cum[d]` is a pd.Series of driver d's simulated race time
    at the end of each lap (index lap), `pit[d]` a pd.Series of the pit loss charged on each lap
    (0 where none), `lengths` durations(). Returns new Series, the input unchanged.
    `held_laps` (the near-lapped rule, ScQueue.held_behind_laps; None: off): a car more than that
    many laps behind the leader when the safety car came out is not a car in the queue; it keeps
    its gap (in the order) to the last car of the train ahead of it; its distance is
    `behind[(S, E)]` (deployment_behind of the recorded race) where given, else its simulated time
    at the end of lap S - 1 behind the first car's over lap_ref_s."""
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
        held = set()
        if held_laps is not None and S > 1:
            b = (behind or {}).get((S, E))
            if b is None and lap_ref_s:
                t1 = pd.Series({d: float(out[d][S - 1]) for d in order})
                b = (t1 - t1.min()) / lap_ref_s
            if b is not None:
                held = {d for d in order if d != lead and float(b.get(d, 0.0)) > held_laps}
        prev = lead
        for d in order[1:]:
            if d in lapped:   # stayed a lap down: not in the lead lap's queue
                continue
            if d in held:     # lapped at the deployment: keeps its gap to the train's last car
                new[d] = new[prev] + float(key[d] - key[prev])
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
                 recorded: bool = False, own_pace_s: float = 0.0, lapped=(), own_lapped: bool = False,
                 held_laps: float = None, placement: str = "clock"):
        self.spacing_s, self.lap_ref_s, self.recorded = float(spacing_s), float(lap_ref_s), recorded
        # the near-lapped rule (ScQueue.held_behind_laps): a car more than held_laps laps behind the
        # leader at the deployment is held (not a car in the queue: it keeps its gap at the
        # deployment to the last car of the train ahead of it). Where the
        # rows show the deployment (deployment_times: the judge's recorded race, a windowed fit's
        # visible rows) the others' keys are their green times at S - 1 from the crossings before it
        # (a recorded crossing of S - 1 made under the safety car would put a car far behind in the
        # queue behind cars it was ahead of) and the distances, the car's own included, are those
        # at the deployment; else the distances come from the times at the end of lap S - 1 (the
        # plan's simulated time for the car, the others' as placed or projected) over lap_ref_s
        self.held_laps = None if held_laps is None else float(held_laps)
        self.rows, self.driver, self._behind = rows, driver, {}
        # placement "record" (ScQueue.placement, the judge only: hindsight): the car is placed in the
        # train by its recorded place at the deployment, a plan moved by its simulated time
        # difference at S - 1 to the plans that stopped where the car really did before S
        self.placement = placement
        self.own_pace_s = float(own_pace_s) if np.isfinite(own_pace_s) else 0.0
        self.own_lapped = bool(own_lapped)
        self.horizon = int(horizon)
        lap1 = rows[rows["lap"] == 1]["start_s"].dropna()
        self.start = float(lap1.min()) if len(lap1) else float(rows["start_s"].min())
        own = rows[rows["driver"] == driver]
        own1 = own.loc[own["lap"] == 1, "start_s"].dropna()
        self.own_offset = float(own1.iloc[0]) - self.start if len(own1) else 0.0
        self.own_t = (own.set_index("lap")["end_s"].astype(float) - self.start)
        self.own_in_laps = {int(L) for L in own.loc[own["pit_in"].astype(bool), "lap"]} if len(own) else set()
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
        asof = self.placement == "asof"
        if asof and (self.held_laps is None or S - 1 > self.horizon):   # a drawn later period: not queued
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
        dep = self.deployment(S) if self.held_laps is not None and S - 1 <= self.horizon else None
        t_key = t_before   # the car's time at S - 1 that places it in the train
        if (((self.placement == "record" and self.recorded) or asof) and dep is not None
                and self.driver in dep[0].index):
            own_eq = float(dep[0][self.driver]) - self.start   # its recorded green time at S - 1
            if S - 1 > k:
                real = np.array([(L in self.own_in_laps) for L in range(first_lap, S)])
                same = ((pit_lap[:, :S - first_lap] > 0) == real[None, :]).all(axis=1)
                if same.any():   # drift of the plans that stopped as the car did, removed from every plan
                    t_key = t_before - (float(np.median(t_before[same])) - own_eq)
            else:
                t_key = np.full(n_plans, own_eq)
        key_d = t_key + pit_lap[:, i0:iE + 1].sum(axis=1)
        if S - 1 <= self.horizon:
            t_o = self.at(S - 1)
            if self.held_laps is not None:   # cars that retired before the period are not in the field
                t_o = t_o[t_o.index.isin(dep[0].index) if dep is not None else self.running(t_o.index, S - 1)]
            if dep is not None:   # the others' green time at S - 1 from their crossings before the deployment
                t_o = t_o.where(~t_o.index.isin(dep[0].index), (dep[0] - self.start).reindex(t_o.index))
            stops = self.pit.reindex(index=range(S, min(E, self.horizon) + 1), columns=t_o.index).fillna(0.0).sum(axis=0)
            key_o = np.broadcast_to((t_o + stops * pit_loss_s * sc_factor).to_numpy(dtype=float)[None, :],
                                    (n_plans, len(t_o)))
        else:   # projected from the horizon: the gaps to the car held, the car's stops apart
            own_h = base if self.horizon == k else float(self.own_t.get(self.horizon, np.nan))
            if not np.isfinite(own_h):
                return zero
            t_h = self.at(self.horizon)
            if self.held_laps is not None:
                t_h = t_h[self.running(t_h.index, self.horizon)]
            gaps = t_h.to_numpy(dtype=float) - own_h
            # the car's time at S - 1 without its stops
            drive = np.cumsum(lap_s, axis=1)[:, S - 1 - first_lap] + base + self.own_pace_s * (S - 1 - k)
            key_o = drive[:, None] + gaps[None, :]
        if key_o.shape[1] == 0:
            return zero
        held_d = np.zeros(n_plans, dtype=bool)
        if self.held_laps is not None:
            if dep is not None:   # the distances at the deployment, the car's own included
                lead0 = float(dep[0].min()) - self.start   # (t_o holds the deployment times where known)
                b_o = np.broadcast_to(((t_o.to_numpy(dtype=float) - lead0) / dep[1])[None, :], key_o.shape)
                if self.driver in dep[0].index:
                    held_d = np.full(n_plans, (float(dep[0][self.driver]) - self.start - lead0) / dep[1] > self.held_laps)
                else:
                    held_d = (t_key - lead0) / dep[1] > self.held_laps
            else:   # from the times at the end of lap S - 1 (others placed or projected, the plan's own)
                t1_o = (np.broadcast_to(t_o.to_numpy(dtype=float)[None, :], key_o.shape) if S - 1 <= self.horizon
                        else key_o)
                lead = np.minimum(t1_o.min(axis=1), t_key)
                b_o = (t1_o - lead[:, None]) / self.lap_ref_s
                held_d = (t_key - lead) / self.lap_ref_s > self.held_laps
            key_o = np.where(b_o > self.held_laps, np.inf, key_o)   # a held car is not a car in the queue
            if not np.isfinite(key_o).any():
                return zero
        first = np.minimum(key_o.min(axis=1), key_d)
        ahead = (key_o < key_d[:, None]).sum(axis=1)
        if held_d.any():   # the train's last car ahead of a held car: its key, and its place in the train
            last_key = np.where(key_o < key_d[:, None], key_o, -np.inf).max(axis=1)
        recorded = self.recorded and all(L in self.sc_laps for L in range(S, E + 1))
        length = self.length(S, E) if recorded else None
        if (self.held_laps is not None and self.recorded and dep is not None and E in self.sc_laps
                and E in self.all_t.index and self.all_t.loc[E].notna().any()):
            # the train reaches the end of lap E when the recorded race's leader did, whatever lap
            # the car's own count saw the safety car from (a car lapped at the deployment counts its
            # laps beside the leader's next ones): measured from the first car's green time at S - 1
            length = float(self.all_t.loc[E].min()) - (float(dep[0].min()) - self.start)
        if length is None:   # the car's own neutralised time from the end of lap S - 1
            length = run[:, iE] - t_before - pit_lap[:, i0:iE + 1].sum(axis=1)
        t_new = first + length + self.spacing_s * ahead
        if held_d.any():   # held: the last car's new time (spacing_s per car ahead of it) plus the gap to it
            gap = np.where(np.isfinite(last_key), key_d - last_key, 0.0)
            t_held = first + length + self.spacing_s * np.maximum(ahead - 1, 0) + gap
            t_new = np.where(held_d, t_held, t_new)
        # where own_pace_s stands in for a pace term the car's clock carries none: the shift lands
        # it at its place in the queue and its later laps run as before
        return t_new - run[:, iE]

    def running(self, names, lap: int) -> np.ndarray:
        """Whether each car was still in the race at the end of `lap` (the near-lapped rule): its rows
        reach within two laps of min(lap, the last lap any row shows). at() projects a car beyond its
        last row (a car behind on the road when a windowed fit's rows were cut), so a car that had
        retired would otherwise be placed in the field, ahead of every car, at lap_ref_s a lap."""
        last = self.t.apply(lambda c: c.last_valid_index()).reindex(list(names)).astype(float)
        upto = min(lap, int(self.t.index.max())) if len(self.t.index) else lap
        return (last >= upto - 2).to_numpy()

    def deployment(self, S: int):
        """deployment_times of the rows for the period starting at S (cached); None where the rows
        do not show the deployment."""
        if S not in self._behind:
            self._behind[S] = deployment_times(self.rows, S, self.lap_ref_s, whole=self.recorded)
        return self._behind[S]


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
