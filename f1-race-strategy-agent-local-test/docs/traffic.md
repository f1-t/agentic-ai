# Ghost-car traffic (SPEC 0012)

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
- `ghost`: traffic on at the default params.
- `ghost_dirty`: information only. This is `ghost` with `dirty_air_s` set to the ghost
  arm's `clean_pack` mean residual (by_kind), the dirty air the clamp leaves unexplained.
  The default `dirty_air_s` stays 0.

Every row is stamped with `pace`, `pit_loss_s`, `pit_loss_source`, `config_hash` (the
arm's own) and `sim_code_version`. The drivers rows of the traffic arm add `traffic_s` (net time
traffic added over the race), `passes` and `conjured_s` (see below). With the default params
the `traffic_ghost.yaml` config hash is `a50205e6c507`. `traffic` is part of every config
hash, so adding the field moved the default hash too (`eccb59070c14`); under the reports
README rule that drift alone regenerates nothing.

In code: `Simulator(frame, SimConfig(traffic={"name": "ghost"}))`, with a snapshot whose
`elapsed_s` is set. `BatchResult.traffic_s` and `.passes` (per plan and sample) and
`SimResult.traffic_s` / `.passes_mean` report what traffic added; `times`, the stint sums
and `compare` include it.

## The rule

Common clock = session time. Over the remaining laps, `G[j, L]` is the end of ghost `j`'s
lap `L`, `G_prev` the end of its previous lap (the lap-k row for the first remaining lap),
`g = G - G_prev` its recorded lap time. Our clock starts at the anchor: the race start (our
driver's lap-1 start, else the field minimum) plus `snapshot.elapsed_s`. Per lap, with `T`
our clock at the lap start, `f` our lap time as the engine produced it and
`arrive = T + f`:

- SC or VSC lap: no traffic.
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
`conjured_s` column sums the negative increments per driver; on Spain 2021 it is -1.55 s
per driver (-29.5 s over 19 finishers, 98 passes).

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
| `dirty_air_range_s` | 2.0 | drivers and commentary put the dirty-air window near 2 s |
| `max_passes_per_lap` | 3 | |

## Known limits

1. **SC bunching.** The SC model is time-based, so a plan 20 s ahead of a ghost before the
   SC is still 20 s ahead after it while the recorded ghosts are bunched; positions on the
   laps after an SC are approximate. A position-based SC is the 0013 candidate; the
   `placement` table (lap 12 on Spain, after the lap 8-10 SC) records the evidence.
2. **Lapped cars.** Ghosts are compared on the same lap number, so a car a lap down is
   "behind" in lap index even when ahead on track; blue flags make this second order.
3. **Ghosts do not react.** No undercut defence, no rival strategist.
4. **Clock drift.** Our clock drifts from the real one by the model error (0.1-0.3 s/lap),
   so positions far from the snapshot are noisy; the rejoin a few laps on is the
   trustworthy use (`calibration.rejoin`).
