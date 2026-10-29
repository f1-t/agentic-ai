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
`table` column (`by_kind`, `drivers`, `pairs`, `gate`, `rejoin`, `placement`, and `sc_laps`
for the bunching arm) and a `traffic` column, with the pace term in the engine in every arm:
- `none`: traffic off.
- `ghost`: traffic on at the default params (SC bunching off).
- `ghost_bunching`: `ghost` with `sc_bunching` true, for the bunching off against on
  comparison. Its `sc_laps` rows give `sc_field_delta_s`, the field's recorded SC lap
  less the engine's, per finisher and SC lap.
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
added; `times`, the stint sums and `compare` include it. With `return_laps`,
`BatchResult.sc_field_delta` and `SimResult.sc_field_delta_s` (sample 0) give the field's
SC lap less the engine's on each bunched SC lap.

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
- SC lap: no traffic, as on a VSC lap (the default); with `sc_bunching` true, the queue
  (below).
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
increments are `bunching_s`); on Spain 2021 it is -1.55 s per driver at the default params
(5.16 passes per driver), and -2.22 s with bunching on (7.11 passes).

**SC bunching (`sc_bunching`, off by default; see the decision below).** Under the SC the
field closes up into a queue about `sc_interval_s` apart, so on an SC lap we take our place
in it instead of keeping the pre-SC time gap.

- **The queue** is the ghosts with both lap ends recorded, not on their in-lap (they leave
  for the pit lane; out-laps rejoin and stay), and within half a reference lap of our
  clock, `|G_prev - T| < 0.5 * ref_lap_s`. That is the same-lap-count guard: a lapped
  car's end of lap `i`, about a lap away in time, is not the car ahead.
- **The lap time is the field's.** `f_field` = the median of `G - G_prev` over the queue's
  ghosts not on their out-lap: the recorded SC lap on this lap, so our clock and the
  ghosts' are comparable. The engine's `f` is the fallback when no ghost qualifies.
- The provisional end is `e = T + f_field + pit loss`. A stop under the SC is booked
  before the placement, so we join the queue where the reduced pit loss puts us. In the
  queue, `A` is directly ahead of `e` (largest `G <= e`) and `B` directly behind
  (smallest `G > e`):

- `A` and `B`: `end = G_A + min(sc_interval_s, (G_B - G_A) / 2)`
- `A` only: `end = G_A + sc_interval_s`
- no ghost ahead (we lead): `end = e`, the lap is `f_field`

The lap time becomes `end - T - pit loss`, so the pit loss stays out of it. We stay
between `A` and `B` (never past `B`) and nobody is passed on an SC lap. The change over
the engine's lap (`f_field - f` plus `end - e`) is reported as `bunching_s`, apart from
`traffic_s`. With bunching on, the engine's SC lap factor matters only with traffic off.
The SC laps are the engine's (`scenario.sc_laps`), which for drawn events already start
inside the snapshot's SC.

The field lap was added after the first version, which used the engine's flat SC lap for
`e`. On Spain 2021 that lap is 114.1 s, while the recorded leader ran laps 8-10 in 85.5,
134.9 and 139.9 s, and the queue at the end of lap 10 is 8.5 s long for 18 cars. The
mismatch moved cars through the queue by up to ten places a lap. With the field lap, the
medians of `sc_field_delta_s` on laps 8, 9 and 10 are -19.4, +3.3 and +17.6 s.

**Decision (0013, fixed in advance).** `sc_bunching` would default on only if, with the
field lap, post-SC placement at lap 12 improved on bunching off (mae 3.00) and the
simulated gate and the pairs were no worse. On Spain 2021 it does not pass:

| | bunching off | bunching on |
|---|---|---|
| placement at lap 12: mae / within_1 | 3.00 / 0.316 | 3.05 / 0.158 |
| simulated gate: within_2 / mae_pos | 0.895 / 1.16 | 0.842 / 1.26 |
| pairs sign agreement | 6 / 8 | 5 / 8 |

So it is off by default and stays available. Under drawn events it is also not a model
of an SC: a drawn SC lap is usually green in the recorded field, so `f_field` is a green
lap, and an SC costs about nothing (the Spain optimum with rate events is about 55 s
faster with bunching on).

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
| `dirty_air_range_s` | 1.5 | `FREE_AIR_GAP_S` (config): `dirty_air_s` is measured on `clean_pack` laps, gaps under that, so it is applied over the same window; see below |
| `max_passes_per_lap` | 3 | |
| `sc_bunching` | false | failed the 0013 decision rule on Spain (above) |
| `sc_interval_s` | 1.2 | about ten car lengths under the SC, 1-1.5 s at SC speed |

The dirty-air window is tied to `FREE_AIR_GAP_S`, one constant. LORO may still select a
wider application window, because the penalty is measured on recorded gaps but applied
on simulated ones. Our clock drifts by 0.1 to 0.3 s per lap, so on Spain 2021 a wider
window (2.0 s) was partly compensating for drift: at `dirty_air_s` 0.229 the clean_pack
residual is 2.5 s per driver with 2.0 s and 2.9 s with 1.5 s. The proper fix is a better
clock.

## Known limits

1. **SC laps keep their time gaps by default.** The SC model is time-based, so a car 20 s
   ahead of a ghost before the SC is still 20 s ahead after it, while the recorded ghosts
   are bunched. `sc_bunching` (above) is the queue rule, available but off: it did not
   improve the post-SC placement on Spain, and under drawn events the recorded field's lap
   on a drawn SC lap is not an SC lap. With it on, the restart is not modelled (a car keeps
   its slot and gains only by the green-lap rule). The `placement` table (lap 12 on Spain,
   after the lap 8-10 SC), its `pit_under_sc` split and `sc_laps` record the evidence.
2. **Lapped cars.** Ghosts are compared on the same lap number, so a car a lap down is
   "behind" in lap index even when ahead on track; blue flags make this second order.
3. **Ghosts do not react.** No undercut defence, no rival strategist.
4. **Clock drift.** Our clock drifts from the real one by the model error (0.1-0.3 s/lap),
   so positions far from the snapshot are noisy; the rejoin a few laps on is the
   trustworthy use (`calibration.rejoin`).
