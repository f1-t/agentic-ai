# Ghost-car traffic (SPEC 0012, SC bunching 0013)

`src/sim/traffic.py` adds a track-position penalty to the simulator: every other driver's
recorded laps are replayed as ghosts on the session clock, and a simulated car that catches
one is held behind it or pays to pass it. The ghosts do not react to us.

## Boundary: world model only

The ghosts' remaining laps are the future of the race. Traffic therefore belongs to the
world model (ground truth, calibration, replay) and never to the agent's belief side. The
agent's tools run with traffic off until a projected-ghost source exists (0012b; the
`projected` source is reserved for it). Traffic is off by default (`SimConfig.traffic`
is `none`), and with it off every seed reproduces the earlier results exactly.

## Turning it on

```yaml
# configs/sim/traffic_ghost.yaml
traffic:
  name: ghost
```

```bash
python -m src.sim.calibrate traffic --season 2021 --gp Spain --pace practice
python -m src.sim.calibrate traffic --config configs/sim/traffic_ghost.yaml   # same, explicit
```

The `traffic` command writes `reports/calibration/<race_id>_traffic.csv`: one csv with a
`table` column (`by_kind`, `drivers`, `pairs`, `gate`, `rejoin`, `placement`) and a
`traffic` column, with the pace term in the engine in every arm:
- `none`: traffic off.
- `ghost_no_bunching`: `ghost` with `sc_bunching` false, the SC laps keeping their time
  gaps (the rule before 0013), for the bunching off against on comparison.
- `ghost`: traffic on at the default params.
- `ghost_dirty`: information only. This is `ghost` with `dirty_air_s` set to the ghost
  arm's `clean_pack` mean residual (by_kind), the dirty air the clamp leaves unexplained.
  The default `dirty_air_s` stays 0.

Every row is stamped with `pace`, `pit_loss_s`, `pit_loss_source`, `config_hash` (the
arm's own), `sim_code_version` and `pace_offset_def`. The drivers rows of the traffic arms add
`traffic_s` (net time traffic added on green laps), `bunching_s` (net time the SC queue
added), `passes` and `conjured_s` (see below). The placement rows add `pit_under_sc` (an
in-lap on an SC lap before the placement lap), and the summaries split the placement on it.
With the default params the `traffic_ghost.yaml` config hash is `a50205e6c507`; the hash
covers the params a config gives, not the defaults, so a changed default moves
`sim_code_version` only. `traffic` is part of every config
hash, so adding the field moved the default hash too (`eccb59070c14`); under the reports
README rule that drift alone regenerates nothing.

In code: `Simulator(frame, SimConfig(traffic={"name": "ghost"}))`, with a snapshot whose
`elapsed_s` is set. `BatchResult.traffic_s`, `.passes` and `.bunching_s` (per plan and
sample) and `SimResult.traffic_s` / `.passes_mean` / `.bunching_s` report what traffic
added; `times`, the stint sums and `compare` include it.

**Pace offsets under traffic.** With traffic on, the drivers table's `pace_offset_s`
(calibration.run) and the midrace `_pace_offset` use the driver's free-air laps, as under
`fit_free_air_only`: the engine already adds the traffic, and an all-laps offset would
count part of it twice. Below `MIN_OFFSET_LAPS` free-air laps they fall back to every lap
and flag `pace_offset_all_laps`. `pace_offset_def` (`free_air` or `all_laps`) labels each
arm; for traffic arms the raw bias, mae and mae_green are the like-for-like comparison.

## The rule

Common clock = session time. Over the remaining laps, `G[j, L]` is the end of ghost `j`'s
lap `L`, `G_prev` the end of its previous lap (the lap-k row for the first remaining lap),
`g = G - G_prev` its recorded lap time. Our clock starts at the anchor: the race start (our
driver's lap-1 start, else the field minimum) plus `snapshot.elapsed_s`. Per lap, with `T`
our clock at the lap start, `f` our lap time as the engine produced it and
`arrive = T + f`:

- VSC lap: no traffic, the gaps are kept.
- SC lap: the queue (below). With `sc_bunching` false, no traffic, as on a VSC lap.
- Otherwise a ghost is *eligible* when we start the lap behind it (`d = T - G_prev > 0`),
  both its lap ends are recorded, and the lap is not its in-lap. We *catch* it when
  `arrive < G + follow_gap_s`. It is *fast* for us when `g - f >= pass_delta_s`
  (`pass_delta_drs_s` with DRS on and `d <= drs_range_s`).
- **Held:** a caught car that is not fast; or a caught car on its out-lap when we arrive
  after its line. `clamp` = the largest `G + follow_gap_s` over held cars.
- **Passing:** a caught, fast car that is not on its out-lap and whose `G - pass_gap_s` is
  not below `clamp`, so a pass never takes us past the car that holds us. With more than
  `max_passes_per_lap` such cars the closest ones (largest `G_prev`) are passed and the
  rest count as held (clamp recomputed, the kept passes rechecked against it).
- `free = arrive + passes * pass_cost_s + dirty`, `swap = min over passed cars of
  G - pass_gap_s`, and `end = max(clamp, min(free, swap))`. The lap time becomes `end - T`.
- Pit loss is booked at the end of the in-lap, so the out-lap starts from the rejoin
  position; it stays out of the lap time, as before.

A pass is a position swap: it always ends at least `pass_gap_s` ahead of every car passed,
so the next lap starts ahead of them and a pass is never charged twice. When `free > swap`
the swap grants time the car did not earn, at most `follow_gap_s + pass_gap_s +
pass_cost_s - (g - f)` on that lap, so a lap's increment can be negative. That is the price
of ghosts that do not react (in a reactive world the passed car would lose that time). The
`conjured_s` column sums the negative increments per driver off the SC laps (the SC queue's
increments are `bunching_s`); on Spain 2021 it is -1.39 s per driver at the default params
(5.74 passes per driver), and -1.55 s with bunching off (5.16 passes).

**SC bunching.** Under the SC the field closes up into a queue about `sc_interval_s`
apart, so on an SC lap we take our place in it instead of keeping the pre-SC time gap.
With `f` the SC lap time as the engine gives it (the same for every car), the provisional
end is `e = T + f + pit loss` on that lap: a stop under the SC is booked before the
placement, so we join the queue where the reduced pit loss puts us. Among the ghosts with
a recorded line crossing on this lap, `A` is directly ahead of `e` (largest `G <= e`) and
`B` directly behind (smallest `G > e`):

- `A` and `B`: `end = G_A + min(sc_interval_s, (G_B - G_A) / 2)`
- `A` only: `end = G_A + sc_interval_s`
- no ghost ahead (we lead): `end = e`, the lap time unchanged

The lap time becomes `end - T - pit loss`, so the pit loss stays out of it. The increment
is negative when we close up to the queue and positive when we were closer to `A` than the
interval. We stay between `A` and `B` (never past `B`) and nobody is passed on an SC lap.
The increments are reported as `bunching_s`, apart from `traffic_s`. The SC laps are the
engine's (`scenario.sc_laps`), which for drawn events already start inside the snapshot's
SC.

The order is kept relative to `e`, which is the engine's time-based SC lap. Over several
SC laps the queue reproduces the recorded one only as far as `f` matches the recorded SC
laps. On Spain 2021 the leader's laps 8-10 are 85.5, 134.9 and 139.9 s against the model's
114.1 s each, while the recorded queue at the end of lap 10 is 8.5 s long for 18 cars, so
the mismatch moves cars by several places in one lap (see the 0013 PR).

A ghost on its out-lap is never passed on track: if we arrive before its line it rejoined
behind us (no cost, no swap); within `follow_gap_s` after its line it holds us (the
undercut victim's reality); later than that, nothing. Its in-lap never holds us.

DRS is off on race laps 1-2, on SC and VSC laps, and for two laps after an SC period ends.
A held car sits at `follow_gap_s = drs_range_s`, which is in DRS range by design.

## Constants

A priori and frozen; all of them will be selected by leave-one-race-out across races
later. The sources are placeholders for that selection, not facts the code depends on.

| param | default | placeholder source |
|---|---|---|
| `follow_gap_s` | 1.0 | drivers and commentary put a stable following gap near 1 s |
| `drs_range_s` | 1.0 | DRS detection uses a 1.0 s gap |
| `pass_delta_s` | 0.8 | Barcelona is hard to pass on, hence 0.8 rather than the 0.5 often quoted for easy tracks |
| `pass_delta_drs_s` | 0.3 | |
| `pass_cost_s` | 0.4 | the order of a lap lost per ten overtakes |
| `pass_gap_s` | 0.2 | a car length and a bit at the line; a pass never ends in a tie |
| `drs_off_first_laps` | 2 | |
| `drs_off_after_sc_laps` | 2 | SC only, not VSC |
| `dirty_air_s` | 0.0 | off by default |
| `dirty_air_range_s` | 1.5 | `FREE_AIR_GAP_S` (config): `dirty_air_s` is measured on `clean_pack` laps, gaps under that, so it is applied over the same window |
| `max_passes_per_lap` | 3 | |
| `sc_bunching` | true | |
| `sc_interval_s` | 1.2 | about ten car lengths under the SC, 1-1.5 s at SC speed |

## Known limits

1. **SC bunching is a queue rule.** It places us in the recorded queue lap by lap (see
   the rule above) and does not model the restart: a car keeps its slot, nobody gains on
   the restart lap beyond the green-lap rule, and lapped cars are queued by lap index like
   everything else here. Its queue position is only as good as the engine's per-lap SC lap
   time against the recorded one. The `placement` table (lap 12 on Spain, after the lap
   8-10 SC) and its `pit_under_sc` split record the evidence.
2. **Lapped cars.** Ghosts are compared on the same lap number, so a car a lap down is
   "behind" in lap index even when ahead on track; blue flags make this second order.
3. **Ghosts do not react.** No undercut defence, no rival strategist.
4. **Clock drift.** Our clock drifts from the real one by the model error (0.1-0.3 s/lap),
   so positions far from the snapshot are noisy; the rejoin a few laps on is the
   trustworthy use (`calibration.rejoin`).
