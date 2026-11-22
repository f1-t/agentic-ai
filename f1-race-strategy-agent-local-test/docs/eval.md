# Evaluation harness

`src/eval` scores any strategy policy against the optimizer from the same snapshot, over a
fixed, versioned set of decision points. No model is needed except for the `agent` policy.

```bash
python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json
python -m src.eval reference --scenarios data/scenarios/v1.json      # optional: cache the references
python -m src.eval score --scenarios data/scenarios/v1.json --policies baselines \
    --recorded data/eval/baseline_agent_sample_run.json --out reports/eval/v1_baselines.csv
python -m src.eval report reports/eval/v1_baselines.csv
```

`score` writes one row per (scenario, policy) to `--out` and the summary table beside it
(`*_summary.csv`), and prints the table. `--policies agent` adds the live agent through
whatever `LLM_BACKEND` is configured (see "The agent policy"). `--belief-config` and
`--world-config` (alias `--config`) choose the two simulators below; `--world-configs a b`
is the optional model-disagreement flag. `--max-stops 0` with the agent or a recorded run
in the policies warns (`NoStopReferenceWarning`: "agent decisions are compared against a
no-stop reference"), so that comparison is a deliberate choice.

## Who sees what

Each race has two simulators over the same lap frame and practice
(`score.race_simulators(profile, belief_config, world_config)` gives
`sim_for(race_id, role)`, role `belief` or `world`):

| role | default config | fit | used by |
| --- | --- | --- | --- |
| belief | `configs/sim/belief.yaml` | `to_snapshot`: race laps up to the snapshot, shrunk toward the practice prior; traffic off; SC and VSC periods from the rate model (EventModel defaults, provisional, Spain-derived); samples carry posterior parameter draws and residual lap noise, drawn from seed + `agent_tools.BELIEF_SEED_OFFSET` so the belief never plans on the world's samples (docs/rules.md R6) | every policy: the baselines plan on it through the bridge (`from_contract(snapshot, belief)`, `pace_for_crossover(belief, snapshot)`), and agent decisions are expanded on it |
| world | the profile's SimConfig (no SC/VSC by default; `configs/sim/world_hybrid.yaml` adds them, "Safety cars and VSC in the scoring world") | `full`: every race lap (hindsight); traffic off; the whole-race pit loss ("Pit loss and track status") | the reference `optimize_batch` and every scoring call |

A policy is called as `policy(scenario, belief, max_stops)` and never sees the world. The
regret definition is unchanged: the policy's plan and the world optimizer's plan, both
scored by the world. Traffic is a switch on each side, off by default ("Traffic" below):
`--world-config configs/sim/world.yaml` scores with the other cars' recorded laps, and
`--belief-config configs/sim/belief_traffic.yaml` lets the policies plan with the field
projected from what is visible at the snapshot. The recorded ghosts are valid only under the
recorded events: an event model other than `none` (or a fixed scenario other than the
record) adds `traffic_recorded_with_drawn_events` to every plan's support warnings
(docs/traffic.md, limit 5), unless `align_neutral` keeps the field's gaps on the laps where
they differ (world.yaml sets it).

Rows carry `belief_config_hash`, `world_config_hash` and `belief_window`. The config hash
leaves `fit_window` out, so `belief_window` says which window the policies planned on.

## Traffic

Other cars can be modelled on both sides by the ghost-car model of `src/sim/traffic.py`
(docs/traffic.md has the rule). It is a switch on each side, **off by default**; the fidelity
scorecard (`python -m src.sim.fidelity`) decides by the finishing-order measures, with
traffic off and on, whether it becomes the default:

| side | off (default) | on | the other cars | what it reads |
| --- | --- | --- | --- | --- |
| world (scorer) | the default world | `--world-config configs/sim/world.yaml`: `ghost`, source `recorded`, `align_neutral`, `own_pace`, `pass_mode: earned`, `dirty_air_auto` | their recorded laps, the whole race | everything: it is the scorer, never a policy |
| belief (policies, tools) | `configs/sim/belief.yaml` | `--belief-config configs/sim/belief_traffic.yaml` (the agent's tools: `AGENT_BELIEF_CONFIG=configs/sim/belief_traffic.yaml`): `ghost`, source `projected`, `own_pace`, `pass_mode: earned` | projected forward from the snapshot | `data.visible_rows` at the snapshot moment only |

Each "on" config is its "off" config plus the traffic block and nothing else
(tests/test_traffic_eval_params.py checks it). `score.WORLD_TRAFFIC_CONFIG` and
`score.BELIEF_TRAFFIC_CONFIG` name them.

What a policy may know about the other cars at the snapshot (driver D after lap L): every
row that had ended when D ended lap L (`data.visible_rows`, the clock the belief window and
the pit-loss estimate use), so a car behind has shown its lap L - 1 but not its lap L, and a
car a lap ahead has shown its lap L + 1: their line crossings (positions and gaps, recomputed
from those rows alone), lap times, compounds, tyre ages and stops, and the field's typical
tyre life at the stops seen so far. It may not know what any of them does afterwards. The
projection (traffic.py "Projected ghosts") starts each car from its own last visible
crossing (a lap in progress ends after the cut): it laps at the belief's fitted lap time for
its compound and age plus its own median residual over its recent green laps; a car that has
not stopped stops once when its set reaches the field's typical life so far (per compound),
at the latest `min_stint_laps` before the flag, and a car that has stopped stops again only
if its set would otherwise pass 1.3 times that life; a projected stop costs the snapshot's
pit-loss prior on the out-lap and fits the compound most used at the stops so far. A car
with no crossing for two field laps before the cut is out of the race. Eight variants (stop
lap +-3 laps, pace by its standard error, at least 0.15 s) are cycled over the samples, which
gives the bands. The recorded source is refused belief-side (`agent_tools.belief_simulator`),
and the engine checks the recorded events only for the recorded source.

No future information: `tests/test_future_blindness.py` has a `traffic` surface (the belief
with traffic on: the projected ghost table of every variant, own pace and dirty air, a plan
set's times and positions, get_model_report, compare_plans and evaluate_strategy as shown,
the belief optimum and run_to_flag planned with it), so its truncate, scramble and
aggregates alterations cover traffic. `tests/test_traffic_future_blindness.py` is the
targeted check on a synthetic race: every other car's rows that ended after the cut are
altered (lap times, clock, compounds, tyre ages, stops, track status, positions, gaps, a
retirement), and the traffic state, batch times and positions, the tools' outputs and every
policy's plan (belief_optimizer, the bridged baselines, the rule baselines; actual_strategy
excepted) come out identical; the recorded source, which reads them, does not.

The agent sees, per plan or action (compare_plans `traffic`, evaluate_strategy `traffic`),
with traffic on; every string goes through `tools.present` like any tool output and names
no season, event or driver. 2021 Spain, the car in P2 after lap 20 (1.5 s behind the
leader), evaluate_strategy of a pit for MEDIUM against staying out a lap (measured before
the as-of changes of #90-#101, numbers illustrative):

```json
{"action": {"expected_rejoin_position": 3.0, "rejoin_position_band": [1.0, 5.0], "cars_in_pit_window": 5.7,
            "time_lost_s": {"low": 0.4, "median": 1.2, "high": 3.0},
            "final_position": {"low": 1.0, "median": 1.0, "high": 1.5}},
 "alternative": {"expected_rejoin_position": 1.0, "rejoin_position_band": [1.0, 2.0], "cars_in_pit_window": 0.2,
                 "time_lost_s": {"low": 0.0, "median": 0.0, "high": 0.8},
                 "final_position": {"low": 1.0, "median": 1.0, "high": 1.5}},
 "basis": "other cars projected from what is known at this lap only (...)"}
```

get_model_report adds `traffic`: `cars_within_pit_loss_behind_now`, `dirty_air_s`,
`own_pace_s` and the basis in words. `time_lost_s` is included in every time the tools
report. With traffic off the tools say traffic is not modelled.

The scored quantity is still race time (`regret_s`), with the time lost in traffic in it
when the world runs traffic. Rows then also carry `final_position_mean`,
`opt_final_position_mean` and `position_regret` (Metrics), information only. Finishing
positions classify the field at the flag (`traffic.flag_classification`): a car whose last
crossing is on or after the leader's last lap took the flag, a lapped one a lap or more
short, and is ranked on its own final lap; under the scheduled race distance a lapped
finisher is no longer counted as a retirement.

Cost: the traffic step reruns per sample only when the samples differ. A world that draws no
events, noise or parameters has identical samples, so the engine reuses the first sample's
traffic result for the rest (engine.simulate_batch); a projected field never does (its
variants differ per sample).

### Traffic: accuracy

The numbers below were measured on main at bc73f6c, before the as-of changes (#90 pit loss,
#95/#97 events, #98 policy view, #101 wear cost); the commands below rerun them, and the
fidelity scorecard calls them.

`python -m src.sim.traffic_accuracy --races <ids> --out reports/calibration/traffic` and
`python -m src.sim.field_replay --races <ids> --out ...` (both refuse held-out races), on the
calibration races 2021_04_spain 2023_07_spain 2023_01_bahrain 2019_10_united_kingdom
2022_02_saudi_arabia 2024_07_emilia_romagna. Committed outputs:
`reports/calibration/traffic/summary.csv`, `g_by_horizon.csv` and
`field_replay_summary.csv`.

Columns: **G** the real remaining plan replayed from every snapshot lap (calibration.midrace),
predicted less real remaining time, horizons 10-49, pace-corrected by one definition for
every arm (the traffic-off arm's offset, `err_common_s`), mae and signed bias in s.
**rejoin** every real stop of a finisher replayed from the lap before the in-lap: position
error and gap to the car ahead after the out-lap (calibration.rejoin), and the position three
laps later. **pack left** the observed pack penalty the arm leaves unexplained (mean residual
of clean laps within 1.5 s of the car ahead less that of clean free-air laps, full-race
replay); **pack model** the model's own estimate (mean traffic increment on those laps less
on free-air laps). **SG bias** stop-gain bias at real stops (moves only with the fit, so it
is the same off and on).

| race | arm | G mae | G bias | rejoin pos mae | rejoin gap mae | rejoin gap bias | rejoin+3 pos mae | pack left | pack model | SG bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019_10_united_kingdom | world off | 16.84 | -10.92 | 1.30 | 5.45 | -2.68 | 1.93 | 0.02 |  | -0.04 |
| 2019_10_united_kingdom | world on (world.yaml) | 17.41 | -8.11 | 1.52 | 5.36 | -2.67 | 2.19 | -0.07 | 0.09 | -0.04 |
| 2019_10_united_kingdom | belief off | 32.04 | 24.23 | 1.70 | 5.57 | -1.43 | 2.44 |  |  | -0.81 |
| 2019_10_united_kingdom | belief on (belief_traffic.yaml) | 32.39 | 24.73 | 1.70 | 5.51 | -1.38 | 2.52 |  |  | -0.81 |
| 2021_04_spain | world off | 10.93 | -7.03 | 0.19 | 3.58 | -1.17 | 0.57 | 0.24 |  | -0.34 |
| 2021_04_spain | world on (world.yaml) | 9.80 | -3.85 | 0.30 | 3.80 | -1.52 | 0.57 | 0.13 | 0.11 | -0.34 |
| 2021_04_spain | belief off | 19.06 | 1.00 | 0.51 | 4.58 | -1.67 | 0.81 |  |  | -1.01 |
| 2021_04_spain | belief on (belief_traffic.yaml) | 19.09 | 1.73 | 0.51 | 4.58 | -1.67 | 0.81 |  |  | -1.01 |
| 2022_02_saudi_arabia | world off | 44.14 | -35.02 | 1.67 | 3.59 | -2.35 | 4.47 | 0.55 |  | 0.31 |
| 2022_02_saudi_arabia | world on (world.yaml) | 44.09 | -32.99 | 1.67 | 3.38 | -2.13 | 4.27 | 0.38 | 0.18 | 0.31 |
| 2022_02_saudi_arabia | belief off | 27.80 | -0.05 | 1.40 | 3.15 | -1.51 | 4.47 |  |  | -0.22 |
| 2022_02_saudi_arabia | belief on (belief_traffic.yaml) | 27.95 | 0.67 | 1.40 | 3.15 | -1.51 | 4.47 |  |  | -0.22 |
| 2023_01_bahrain | world off | 20.09 | 13.67 | 0.71 | 4.10 | -0.57 | 1.00 | 0.31 |  | -0.44 |
| 2023_01_bahrain | world on (world.yaml) | 21.23 | 15.82 | 0.69 | 4.09 | -0.69 | 0.98 | 0.21 | 0.10 | -0.44 |
| 2023_01_bahrain | belief off | 35.15 | 30.92 | 0.76 | 5.33 | -1.06 | 1.02 |  |  | -0.15 |
| 2023_01_bahrain | belief on (belief_traffic.yaml) | 36.03 | 31.90 | 0.78 | 5.54 | -1.31 | 1.00 |  |  | -0.15 |
| 2023_07_spain | world off | 8.07 | -1.56 | 0.19 | 1.94 | -0.19 | 0.47 | 0.11 |  | -0.53 |
| 2023_07_spain | world on (world.yaml) | 8.09 | -0.09 | 0.21 | 1.94 | -0.25 | 0.53 | 0.01 | 0.10 | -0.53 |
| 2023_07_spain | belief off | 11.64 | 7.24 | 0.26 | 1.94 | 0.10 | 0.47 |  |  | 0.02 |
| 2023_07_spain | belief on (belief_traffic.yaml) | 12.02 | 8.07 | 0.28 | 1.94 | 0.17 | 0.47 |  |  | 0.02 |
| 2024_07_emilia_romagna | world off | 13.27 | -12.09 | 0.17 | 2.20 | 0.93 | 0.50 | 0.21 |  | -0.13 |
| 2024_07_emilia_romagna | world on (world.yaml) | 11.03 | -9.05 | 0.22 | 2.11 | 1.06 | 0.59 | 0.03 | 0.18 | -0.13 |
| 2024_07_emilia_romagna | belief off | 17.32 | 1.56 | 0.17 | 1.81 | 0.65 | 0.45 |  |  | 0.36 |
| 2024_07_emilia_romagna | belief on (belief_traffic.yaml) | 17.27 | 2.31 | 0.17 | 1.80 | 0.83 | 0.45 |  |  | 0.36 |
| **mean** | world off | 18.89 | -8.83 | 0.70 | 3.48 | -1.01 | 1.49 | 0.24 |  | -0.20 |
| **mean** | world on (world.yaml) | 18.61 | -6.38 | 0.77 | 3.45 | -1.03 | 1.52 | 0.12 | 0.13 | -0.20 |
| **mean** | belief off | 23.83 | 10.82 | 0.80 | 3.73 | -0.82 | 1.61 |  |  | -0.30 |
| **mean** | belief on (belief_traffic.yaml) | 24.13 | 11.57 | 0.81 | 3.75 | -0.81 | 1.62 |  |  | -0.30 |

Every arm, race means (summary.csv has the rest):

| arm | G mae | G bias | rejoin pos mae | rejoin+3 pos mae | pack left | pack model |
| --- | --- | --- | --- | --- | --- | --- |
| world off | 18.89 | -8.83 | 0.70 | 1.49 | 0.24 | |
| world, ghost as it was (swap passes) | 19.13 | -8.66 | 0.71 | 1.52 | 0.19 | 0.06 |
| world, + own pace | 19.01 | -8.63 | 0.73 | 1.55 | 0.20 | 0.04 |
| world, + earned passes, no dirty air | 18.87 | -7.79 | 0.75 | 1.53 | 0.16 | 0.09 |
| **world, + dirty air from the race (world.yaml)** | **18.61** | **-6.38** | 0.77 | 1.52 | 0.12 | 0.13 |
| world, the same with the free-air fit | 19.10 | -7.80 | 0.76 | 1.51 | 0.17 | 0.13 |
| belief off | 23.83 | 10.82 | 0.80 | 1.61 | | |
| belief, projected, swap passes | 23.81 | 10.73 | 0.81 | 1.62 | | |
| **belief, projected, earned passes (belief_traffic.yaml)** | 24.13 | 11.57 | 0.81 | 1.62 | | |
| belief, projected, earned, as-of dirty air | 25.12 | 13.78 | 0.82 | 1.70 | | |

**Whole-field replay** (`src/sim/field_replay.py`): every classified car's real strategy
replayed together from lap k to the flag (with traffic on, each car against the other
cars' simulated trajectories after k, two passes), against the real race. Fit
leave-one-driver-out (the full-window fit without car D's laps predicts D); pace term in the
engine: D's race pace up to k (`asof`; this weekend's practice at k = 0). Neither reads D's
laps after k, so no number below is in-sample for the car it predicts; the world fit still
reads the other cars' whole race, as the scorer does. Race means (per race in
`field_replay_summary.csv`; the `practice` pace rows are there too):

| k | time mae off / on | time bias off / on | Spearman off / on | pos mae off / on | within 1 off / on |
| --- | --- | --- | --- | --- | --- |
| 0 | 37.80 / 35.23 | -22.61 / -10.40 | 0.75 / 0.75 | 2.27 / 2.35 | 0.51 / 0.47 |
| 10 | 35.29 / 34.05 | -16.94 / -12.73 | 0.94 / 0.94 | 1.21 / 1.21 | 0.67 / 0.69 |
| 20 | 19.83 / 19.66 | -4.52 / -0.50 | 0.94 / 0.94 | 1.18 / 1.18 | 0.69 / 0.70 |
| 30 | 14.57 / 14.22 | -2.09 / 0.11 | 0.95 / 0.95 | 1.06 / 1.01 | 0.76 / 0.77 |
| 40 | 9.64 / 8.94 | -4.26 / -2.80 | 0.96 / 0.96 | 0.80 / 0.72 | 0.82 / 0.86 |

**What the measures said (before the merge).** The world, on against off: on lowers the G error and bias, the whole-field
time error and bias at every calibration point, and explains half the observed pack
penalty (0.24 s left off, 0.12 s on; the model's own estimate 0.13 s); finishing order is
equal or better from lap 10 on and slightly worse from the start. It is worse on the
rejoin position (0.70 to 0.77 places after the out-lap) and on two races' G (UK, Bahrain,
where the world is already slow). Two parts were fixed rather than switched off: the old
swap rule conjured time from every pass (net traffic time was negative), replaced by
earned passes; and the dirty-air term, off before, is identified from the race's own laps
(0.37 s a lap on average). The free-air fit was not kept (worse G). The belief, traffic on
(the projected field, earned passes, no dirty air): by its own as-of measures it is
not more accurate: G 24.13 against 23.83 s with a larger positive bias (the belief is
already 11 s slow over these horizons, from causes outside traffic), rejoin equal. The
as-of dirty-air estimate is not identified (0.68 s against the race's 0.37 s) and made it
worse (25.12), so it is off. Belief traffic gives the belief the same
physics as the world and the tools can report rejoin positions honestly; the swap rule
would score 23.81 but only by conjuring time. If the belief's bias is fixed elsewhere this
should be measured again. Neither side is on by default now: the fidelity scorecard decides by
the finishing-order measures.

## Information contract

A policy decides at a snapshot: race R, driver D, after lap L. Let t be the session time at
which D completed lap L. **Everything a policy can read must be a deterministic function only
of**

- (a) the raw timing records of race R with timestamps at or before t;
- (b) this weekend's sessions before the race start;
- (c) published regulations and the tyre allocation;
- (d) artifacts built only from races whose start precedes R's start, each with a committed
  manifest of its source races.

"What a policy can read" is the snapshot and the agent's view, every tool output and error
string, the belief simulator's fit, draws, constants and rules, the continuation behind an
agent decision, and every baseline's input. The world simulator and the hindsight reference
are the scorer: they keep the whole race.

`tests/test_eval_information_contract.py` checks it. It builds one synthetic race three ways,
each as `lap_frame` would (the whole-race derived columns recomputed on the rows given): as
run, truncated at t, and with everything after t altered (lap times, compounds including rain
to come, tyre ages, stops, track status, positions and clock, a retirement, the snapshot
driver lapped, a longer race). The snapshot, the anonymised view, the belief's fit, every
agent tool's output and error strings, `belief_optimum`, `to_plan` (the continuation), the
bridged baselines, the rule baselines and the sampler's decision points up to t must be
identical across the three. Each legacy switch below is shown to fail the same check on its
own. The R8 stint limit (item 1 of the leakage audit) is another branch's; this file runs the
belief with R8 off and a fixed RaceParams pit loss (the pit loss is "Pit loss and track
status" below). `tests/test_future_blindness.py` (the future-blindness harness: real races
through `jobs.load_race`'s own code path, the past cut at the snapshot moment, the rest
truncated or re-raced) keeps `frame_filter`, `lap_window`, `gap_behind`, `race_length` and
`compound_list` as regression checks (`future_blindness.FIXED_LEAKS`), observes snapshots from
`snapshot_as_of`, and checks that a snapshot builds from the past alone, that scenario
selection up to the moment is unchanged and that no real identity reaches the agent.

What each fix does (the legacy switch reproduces the old behaviour, so old results stay
reproducible):

| item | was | now | legacy switch |
| --- | --- | --- | --- |
| 3 clean flag and the other derived columns | the frame's `clean`, `pace_resid_s`, `driver_offset_s` and gap columns, from a robust filter fitted on every lap of the race, chose the belief's fit rows, `ref_lap_s` (the SC lap) and the lap-noise sd | `data.as_of_columns` recomputes them on the rows the window can see (`Simulator.window_rows`); the world keeps the whole-race ones | `SimConfig.policy_view.clean: race` |
| 6 window cut by lap number | `lap <= L` for the fit and the calibrator's `laps_so_far`: another car's lap L that ended up to minutes later was in | `end_s <= t` (`Simulator.cut_time`: D's lap-L end; a car a lap ahead's lap L + 1 that ended earlier is in) | `SimConfig.policy_view.window_cut: lap` |
| 4 race length | `snapshot_at` set `total_laps` to D's own lap count: a lapped or retiring car showed it | `calibration.snapshot_as_of` keeps the scheduled distance (`session.total_laps`) | the builder `snapshot_at` |
| 5 compounds | `RaceParams.compounds`: every compound raced in the whole race (2021 Spain showed MEDIUM, SOFT; a wet race would show INTERMEDIATE) | the nominated dry three, HARD, MEDIUM and SOFT (`calibrate.race_compounds`), plus any compound a car had run by t (a wet start) | `calibrate --compounds raced`, `calibrate.DEFAULT_COMPOUND_SOURCE = "raced"` |
| 6 snapshot reads | positions, gaps and the car behind came from the lap L + 1 rows | `snapshot_as_of` reads rows completed by t: the car ahead and its gap at the end of lap L, the car behind at the previous crossing (the last one it had made by t), each car's position at its last lap by t, the stint history from D's laps up to L; on an in-lap the car shows on the set it came in on (no record by t holds the set going on) | `snapshot_at` |
| 7 race identity | the view showed season, event slug and real driver codes | the model sees `"event": "anonymised race"`, no season, its own car as "Car A" and the nearby cars as "Car B", "Car C", ...; `tools.present` applies the same mapping, plus any other code of the field to "another car" and any year to `<year>`, to every tool output and error string. Lap count and pit loss stay (the tools give them). The mapping stays in the harness: `tools.identity()`, `run_agent`'s `stats["identity"]` and the agent row's `meta["identity"]`; scoring and the trajectory file names keep the real ids, the trajectory events hold what the model saw | `AGENT_SHOW_IDENTITY=1` (ablation only, default off) |
| S scenario selection | finishers at most two laps down only, and the lap before each SC or VSC period | `scenarios --rule as_of` (the default for new sets): every driver still running at the decision lap (laps 1..L recorded), at a fixed grid (every 5th lap to the scheduled distance less 10; `neutralised=True` adds the laps D's own record shows a neutralisation in progress or just ended, off by default because a car behind decides after the others at the same lap number), capped to leave one minimum stint of the scheduled distance | `scenarios --rule legacy` |

`SimConfig.policy_view` is left out of `config_hash` while it holds its default, so a config
that does not set it keeps its hash; the behaviour change under the default shows in
`sim_code_version`. Sets sampled before these fixes (v1, pilot1, d2, d3, d4, dev6) have no
`sampling["rule"]`: they were sampled under the legacy rule with `snapshot_at`, and their
snapshots embed the driver's own lap count and the raced compounds. They load and score as
before; a new set records `"rule": "as_of"` and the builder `snapshot_as_of`.

### Cross-race tables, as of the race

Everything the policy side reads that was estimated from other races records the races it was
fitted on and, pricing race R, uses only races that started strictly before R
(`race_ids.as_of`); with no earlier race it falls back to a documented a priori constant with a
wide sd, never fails and never uses a later race. `src/sim/provenance.py` lists every such
artifact with its kind and, per race, the races it reads; `tests/future_blindness.py` checks it
for every real race, 2021 Bahrain, 2019 United Kingdom and a synthetic race dated 1990
(`test_policy_side_artifacts_fitted_only_on_earlier_races`, `test_race_before_every_table_uses_documented_priors`).
The tables are built from the calibration races only (`compound_steps.CALIBRATION_RACES`: the six
development races, 2025 Qatar and thirteen 2021-22 weekends, 2022 Austria among them since its
unlabelled-compound laps load as `UNKNOWN`; the pit-loss prior table holds the nineteen of them
`dev20` samples, 2025 Qatar not); no race of `data/eval_set.json` is ever loaded or feeds a table.

| artifact | read as of the race | no earlier race (a priori) | legacy switch |
| --- | --- | --- | --- |
| proxy tier 5 (`configs/sim/compound_steps.json`) | `per_race` step rows and the forward-chained `validation_rows` (each race priced with the table of the races before it: `compound_steps validate --as-of`); `proxy.as_of_config` recomputes the steps (median), the blend weights (1 / mse per source, at least 3 rows) and the sd floor (the blend's rmse) | no step, equal weights, `proxy.PRIOR_SD_FLOOR` (pace 1.0 s, wear 0.1 s/lap) | `compound_proxy.as_of: false` |
| base delta, fuel prior, SC lap factor, lap-1 extra (`configs/sim/race_history.csv`, `python -m src.sim.race_history build`) | the `as_of_history` calibrator (belief.yaml): median of the earlier races, sd their spread floored at `race_history.SD_FLOOR`; the fuel prior is the anchored prior ("Fuel anchoring": the lap-scaled physics slope times the earlier races' target / physics ratio; `fuel_anchor.prior: circuit_history` gives `fuel_prior`'s circuit history, #45); the sds are the base and fuel taus, the event factors are drawn per sample | `race_history.A_PRIORI`: base delta 0.5 ± 1.0 s, SC lap factor 1.4 ± 0.15, lap-1 extra 8 ± 4 s; the fuel prior is the physics prior (`fuel_practice_longrun`, -0.03 s/kg x 110 kg / laps) | `calibrator: shrinkage` (the Spain-derived config values) |
| SC/VSC rates (`configs/sim/neutralised_history.csv`) | `events.rate` with `as_of: true`: `event_rates`' global posterior on the earlier dry races (#45's forward-chaining fits) | `race_history.EVENT_PRIOR`: p_sc 1/60 per lap, d_sc 3.5 laps; VSC 0.02, 2 laps | `as_of: false` (EventModel's Spain values) |
| `calibration_error_s` (`reports/calibration/midrace_summary.csv`, measured on 2021 Spain) | the summary's value for a race after 2021 Spain | NaN (the legacy `within_sim_error`, `AGENT_GAP_YARDSTICK=calibration`, falls back to the sampling spread; the default paired gap reading never reads it) | `SimConfig.cross_race_as_of: false` |
| pit-loss prior, tyre-life history | as-of already (`pit_loss.pit_loss_prior`, `cliff.life_prior` / `tyre_history`) | their constant tiers | their own |
| shrinkage taus | a priori engineering priors (base 0.5 s, wear 0.03 s/lap, hinge), fitted on no race; under `as_of_history` the base and fuel taus are the as-of sds | | |
| learned calibrator (`trained`) | not as of the race: fitted on a fixed race set, off by default; refused for a race not after all its training races unless `force_not_as_of` | | |

The world (the judge and the hindsight reference) keeps the whole-set tables: its full window
prices unraced compounds with the whole-set step table and runs the config's effects and rates.
The step table, weights and sd floor were re-estimated on the twenty races (the whole-set values
the world and the legacy switch read; the weights and sd floor in the file are still the earlier
seven-race validation's).

**Scoring a car that was lapped or later retired.** Every plan covers the scheduled
distance, the reference's included, and the world scores all of it with its whole-race fit
(the world has no per-car pace and draws no retirement). Policy and reference then cover the
same laps under the same draws, so regret stays a like-for-like difference. Truncating to the
laps the car really ran would need the scorer to cut every plan, the reference's too, at a
number no policy may know, and a retired car has no such number at all. The
`actual_strategy` comparator runs its last recorded set on to the flag when the car covered
fewer laps.
## Pit loss and track status

**The information contract for pit loss.** At (race R, driver D, lap L) a policy may know the
pit loss of earlier races and the stops made in R whose out-lap ended by the time D ended lap L.
It may not know the race's whole-race value, which is measured from stops that have not
happened yet.

| who | pit loss | where it comes from |
| --- | --- | --- |
| world (scorer, reference) | the whole-race value: the median green stop of R (in- plus out-lap time less the full-window fit's lap times), the prior when R has fewer than 3 green stops | `score.world_pit_loss` from the loaded race (`sim.pit_loss.whole_race`), passed as `Simulator.race_pit_loss_s`; never read from a snapshot |
| belief, agent tools, baselines, belief optimizer | the as-of estimate at L, with an sd drawn per sample (engine `PIT_LOSS_STREAM`) | `SimConfig.pit_loss.source: asof` (`configs/sim/belief.yaml`), `sim.pit_loss.estimate` on the belief's fit at L |
| snapshot contract (`calibration.snapshot_at`) | the pre-race prior only | `calibration.pre_race_pit_loss`; frozen sets built before this carry the old whole-race value, so `score.policy_scenario` replaces it with the belief's as-of value before any policy sees the scenario |

The as-of estimate: a prior, then this race's green stops (in- and out-lap both green) seen by
the snapshot, each stop's loss against the belief's own fit, stops outside 0.5-1.5 times the
prior left out, summarised by their median with a per-stop sd max(MAD, 2.5 s); the interval
adds 1.5 s in quadrature for the gap between a per-stop and a whole-race number. The prior is
a fallback chain, as of R's date and never R itself:

1. `same_circuit_asof`: earlier seasons at the same circuit (sd 1.5 s);
2. `cross_circuit_asof`: at least two earlier races at any circuit (sd their spread, at least 2 s);
3. `constant`: 22 s, sd 5 s (no earlier race at all, or a synthetic race id).

The pool is `configs/sim/pit_loss_prior.json` (the six calibration races, built by `python -m
src.sim.pit_loss build-prior`) plus every race in the FastF1 cache at run time (`real_run.cached_races`),
whose entry is built from its whole race on first use and kept per process (a circuit with no history
loads every earlier cached race once, about a minute): more cached seasons widen the pool with no
code change. A wet race (any INTERMEDIATE or WET lap) is left out of the pool. A held-out race (data/eval_set.json) joins the pool only when the race being priced is
itself held out, and then only if earlier. A circuit used once (no earlier season cached) gets tier
2, or 3, and the estimate exists from lap 0 either way: a policy never fails for want of history and
never falls back to the whole-race value. The SC and VSC factors (the share of a green stop paid
when the in-lap is under SC or VSC) start from 0.5 and 0.6 (sd 0.2, worth 5 stops) and are updated
with this race's SC / VSC stops seen by then, each measured against the field's median lap on
those laps. `estimate_pit_loss` reports the as-of range and its basis (prior, prior tier, stops
seen), the factor for the current status and `pit_loss_now_s`, how many laps the current SC or VSC
has lasted, and the chance of a new SC or VSC in the next 5 and 10 laps (below).

**The world's value changed** from the replay mean (`calibrate._pit_loss`, still the G replays')
to the median green stop, which a few slow stops or a mostly neutralised race cannot drag:

| race | replay mean (old world) | median green stop (world) | green stops |
| --- | --- | --- | --- |
| 2019_10_united_kingdom | 24.61 | 21.21 | 20 |
| 2021_04_spain | 23.85 | 22.57 | 34 |
| 2022_02_saudi_arabia | 15.05 | 24.39 | 5 |
| 2023_01_bahrain | 26.59 | 25.14 | 44 |
| 2023_07_spain | 24.94 | 23.77 | 42 |
| 2024_07_emilia_romagna | 28.46 | 28.67 | 27 |

**As-of error** against the world's value, mean signed error (s) by snapshot lap, one snapshot per
5 laps (the driver at the median position), `python -m src.sim.pit_loss report`
(reports/calibration/pit_loss/asof_error.csv):

| race | prior tier (prior s) | laps 0-9 | 10-19 | 20-29 | 30-39 | 40+ | MAE | 90% interval covers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019_10_united_kingdom | constant (22.00) | +0.79 | -0.10 | -1.09 | -0.74 | -0.71 | 0.84 | 1.00 |
| 2021_04_spain | cross_circuit_asof (22.36) | -0.21 | -0.21 | +0.51 | -0.16 | +0.02 | 0.26 | 1.00 |
| 2022_02_saudi_arabia | same_circuit_asof (23.00) | -1.38 | -1.54 | -0.96 | -1.38 | -1.38 | 1.33 | 1.00 |
| 2023_01_bahrain | same_circuit_asof (24.13) | -1.01 | -0.71 | -0.04 | +0.01 | -0.35 | 0.47 | 1.00 |
| 2023_07_spain | same_circuit_asof (22.90) | -0.87 | -0.23 | -0.07 | +0.01 | +0.07 | 0.22 | 1.00 |
| 2024_07_emilia_romagna | cross_circuit_asof (23.24) | -5.43 | -1.68 | -1.60 | -1.25 | -0.92 | 1.97 | 0.83 |

Over all 63 snapshots the MAE is 0.82 s (0.60 s from lap 20), and the 90% interval covers the
world's value at 97% of them. The pool here is the committed table plus the cache as of main
3cf9649 (the 2021-22 weekends of #92 included; wet races are left out of the pool: 2022 Imola, run
partly on intermediates, has a 38 s median green stop, and 2021 Imola does not load). The error is
the prior's until the first green stops (Imola 2024's cross-circuit prior is 5 s short of a long pit
lane), then about a second or less. Saudi Arabia keeps a -1.4 s bias: by lap 45 only 4 green stops
are in.

**Test.** `tests/test_sim_pit_loss.py`: altering every stop after the snapshot (moved, added,
slowed, an SC added) leaves the belief's pit loss, `estimate_pit_loss`, `simulate_plan`,
`evaluate_action` and the belief optimizer's plan unchanged at 2021 Spain HAM lap 30; slowing
the stops already made moves the estimate; the fallback chain's tiers and widths; a circuit with
no history. The future-blindness harness's `test_leak_pit_loss` (tests/test_future_blindness.py)
passes: `pit_loss` moved from `KNOWN_LEAKS` to `FIXED_LEAKS`.

### Safety cars and VSC in the scoring world

Before, the world drew no SC or VSC: a snapshot taken under a safety car was scored as if the
race went green at once, and a stop there cost a full green pit loss, so "pit now, it is cheap"
could never be right in the scorer, while the belief priced it at half.

**Default: off.** The default world still draws no SC or VSC (`score.WORLD_EVENTS_DEFAULT` False);
the world below is opt-in with `--world-config configs/sim/world_hybrid.yaml`. Whether it becomes the
default is decided at integration by the fidelity scorecard (PR #93, `python -m src.sim.fidelity`):
the replay accuracy below supports recorded neutralised laps, but drawing the later periods is a
scoring-design choice that no accuracy measure here validates.

**What the opt-in world models** (`score.WORLD_EVENTS`, events `hybrid`, src/sim/events.py):

- the SC or VSC in progress at the snapshot ends where it really ended: the driver's own recorded
  laps of that kind from the next lap on (the scorer may know them; the policy knows only that it
  is in progress and for how long);
- on those laps the lap time is the field's median recorded lap on that lap (non-pit laps of that
  status), not `sc_lap_factor` x the median green lap; a stop whose in-lap is under SC pays
  `sc_pit_loss_factor` (0.5) of the green pit loss, under VSC `vsc_pit_loss_factor` (0.6);
- every later period is drawn by the rate model from the lap after, on the events substream, so
  every policy's plan and the reference see the same draws for a seed (common random numbers),
  and `regret_se_s` carries the Monte Carlo error. The full record is not used past the period
  in progress: a reference that waits for a safety car no policy could know about would score
  luck, not strategy. A green snapshot draws exactly as `rate` does.

Events `recorded` (every recorded SC/VSC lap, for replays) and `rate` stay available.
`configs/sim/world.yaml` names the same events with ghost traffic.

**Accuracy decides the lap times and the factors**, on the six calibration races only
(`python -m src.sim.neutral_report --races ...`, reports/calibration/neutralised): every
finisher's real strategy replayed on the world (full fit, the race's replay-mean pit loss) from
every G snapshot, with neutralisations off (the old world), on with the lap factors (`on_factor`),
on with the recorded lap times (`on_recorded`), and on with the recorded lap times and SC/VSC
pit-loss factors pooled from the other races' stops (`on_recorded_fitted`, out of sample).
G is the offset-adjusted remaining-time error (`err_offset_s`), MAE / signed bias in seconds;
"SC stops" is the in- plus out-lap time plus the stop's cost against the real one, over every
finisher's stop with its in-lap under SC/VSC; "neutral laps" is the lap-time error on non-pit
SC/VSC laps. 2023 Spain and 2024 Imola have no SC or VSC and do not change.

| race | arm | G MAE / bias | SC/VSC stops MAE / bias (n) | neutral laps MAE / bias (n) |
| --- | --- | --- | --- | --- |
| 2019_10_united_kingdom | off | 64.22 / -62.49 | 50.88 / -50.88 (9) | 34.13 / -34.13 (50) |
| | on_factor | 16.84 / -10.92 | 7.94 / +1.84 | 10.03 / -1.56 |
| | **on_recorded** | **16.52 / -9.65** | **7.42 / +2.23** | **6.96 / -0.59** |
| | on_recorded_fitted | 16.66 / -8.57 | 10.20 / +7.72 | 6.96 / -0.59 |
| 2021_04_spain | off | 19.97 / -14.77 | 49.98 / -49.98 (3) | 29.19 / -29.19 (51) |
| | on_factor | 13.79 / -7.74 | 9.39 / -3.16 | 14.81 / -0.11 |
| | **on_recorded** | **13.71 / -7.62** | 28.64 / +5.34 | **4.31 / -0.08** |
| | on_recorded_fitted | 13.73 / -7.57 | 30.00 / +9.41 | 4.31 / -0.08 |
| 2022_02_saudi_arabia | off | 163.76 / -163.62 | 73.28 / -73.28 (7 SC), 34.47 (1 VSC) | 53.19 / -53.19 (54 SC), 24.03 (44 VSC) |
| | on_factor | 41.06 / -32.85 | 13.85 / -13.85, 12.38 | 20.55 / -19.39, 8.53 |
| | **on_recorded** | **18.99 / -17.58** | 15.30 / -15.30, 21.29 | **4.49 / -0.24, 3.81** |
| | on_recorded_fitted | 19.55 / -18.19 | 16.37 / -16.37, 29.16 | 4.49 / -0.24, 3.81 |
| 2023_01_bahrain | off | 24.31 / -21.98 | 18.99 / -18.99 (6 VSC) | 18.31 / -18.31 (18 VSC) |
| | on_factor | 20.09 / +13.67 | 28.66 / +28.66 | 11.12 / +10.85 |
| | **on_recorded** | **15.33 / +3.63** | **14.00 / +14.00** | **5.57 / +2.75** |
| | on_recorded_fitted | 17.91 / +6.25 | 24.64 / +24.64 | 5.57 / +2.75 |

G by horizon (MAE, s) where it moves, off against on_recorded: 2019 UK 30-39 laps to go 132.8 ->
27.1, 40-49 136.5 -> 24.6; 2021 Spain 50-59 31.0 -> 18.8, 60+ 112.5 -> 25.0; 2022 Saudi 10-19 59.3
-> 13.6, 30-39 272.5 -> 23.4, 40-49 348.0 -> 34.0; 2023 Bahrain 20-29 32.2 -> 15.0, 40-49 25.3 -> 29.6
(the one bucket that gets worse: the VSC is over-priced there, bias +16.8). Full tables:
reports/calibration/neutralised/g_by_horizon.csv.

What was kept and what was left off:

- **Kept: the neutralised laps in the world, with the field's recorded lap times.** Off is far
  worse wherever a race had an SC or VSC; recorded lap times beat the lap factors on G at every race
  and cut the neutral-lap error from 10-21 s to 4-7 s.
- **Kept: the a priori pit-loss factors 0.5 (SC) and 0.6 (VSC).** Pooled from the other races'
  stops they lost out of sample at every race (UK 10.20 against 7.42, Spain 30.00 against 28.64,
  Bahrain VSC 24.64 against 14.00). The stops are few and mixed: Spain 2021's three SC stops
  straddle the deployment lap (factors -0.6 to 2.0), Saudi's nine were nearly all on the
  deployment lap. Measured per race (configs/sim/pit_loss_prior.json): UK about 0.41 (9 stops),
  Spain 0.28 (3), Saudi 1.4 (9, deployment lap), Bahrain VSC 0.07 (6, against the slow VSC lap).
  The world fits no factor per race.
- **Left off: SC bunching** (`traffic.sc_bunching`). It failed its own decision rule
  (docs/traffic.md: placement at lap 12 mae 3.05 against 3.00 off, gate and pairs worse), and it
  needs ghost traffic, which the default world does not run.
- **Kept, by design rather than by accuracy: later periods drawn, not recorded.** Replay accuracy
  is highest with every recorded lap (the table), but scoring a decision against neutralisations
  nobody could foresee would reward luck. The rates are the belief's (`EventModel` defaults); the
  scorer may know more, but no as-of or per-circuit rate table is on the eval path yet (below).
- **The world's pit loss** is the median green stop (above). With it the G replays' MAE over the
  six races is 14.55 s against 14.65 s with the replay mean, mixed per race (UK 17.09 against 16.52,
  Spain 2021 14.27 against 13.71, Saudi 17.41 against 18.99, Bahrain 15.11 against 15.33), and the
  SC-stop error falls at UK (6.86) and Saudi (10.63)
  (reports/calibration/neutralised/world_pit_loss_median). The G replays keep the replay mean.

**What the belief knows.** At lap L: the track status up to L (`snapshot.sc_laps`, `vsc_laps`,
`track_status`), so "under a safety car now" and how long it has lasted; later neutralisations
only as rates. Under SC or VSC the rate model starts inside the current period with a geometric
remaining length (memoryless: how long it has lasted does not shorten it). The reduced pit loss
now is the as-of SC/VSC factor times the as-of green pit loss.

**Event rates: what the evaluated configuration uses.** `configs/sim/belief.yaml` draws with the
`EventModel` defaults (p_sc 0.015 per green lap and 3-lap periods from the one SC of 2021 Spain;
the VSC rate 0.02 and 2-lap periods are placeholders). No scenario carries per-race rates
(`sc_rate_per_lap` is null in every committed set), so PR #45's forward-chaining event-rate folds
(`event_rates.EventRatesCalibrator`) are not on the eval path, and no held-out race feeds the
rates. They are not as-of, though: for 2021 Spain they are that race's own SC (one period), and for
the races before May 2021 (2019 UK; held-out 2019 Monaco and 2020 Sakhir) they come from a later
race. Not fixed here: an as-of table needs the neutralised periods of earlier races, which the
cache holds for too few seasons; the tools say "provisional" in `rates_source`.

**What the agent sees.** `estimate_pit_loss` under the 2019 UK safety car (HUL, lap 20, as-of
belief):

```json
{"pit_loss_s": {"low": 17.27, "median": 19.99, "high": 22.72}, "source": "asof", "sd_s": 1.66,
 "pit_loss_now_s": {"low": 8.63, "median": 10.0, "high": 11.36}, "pit_loss_now_factor": 0.5,
 "sc_pit_loss_factor": 0.5, "vsc_pit_loss_factor": 0.6, "factor_source": "asof",
 "track_status": "sc", "neutralised_for_laps": 1,
 "outlook": {"next_5_laps": {"sc": {"low": 0.037, "median": 0.073, "high": 0.141},
                             "vsc": {"low": 0.049, "median": 0.096, "high": 0.185}},
             "next_10_laps": {"sc": {"low": 0.073, "median": 0.14, "high": 0.263}, "vsc": {...}},
             "mean_period_laps": {"sc": 3.0, "vsc": 2.0},
             "rates_source": "provisional: SC from 2021 Spain, VSC placeholder", "band": "rate / 2 to rate x 2"},
 "basis": {"prior_s": 22.0, "prior_source": "constant", "green_stops_seen": 12, "sc_stops_seen": 0,
           "vsc_stops_seen": 0, "sc_factor_sd": 0.089, "vsc_factor_sd": 0.089}}
```

(numbers rounded here). The outlook's band is a convention, the rate halved and doubled, since the
provisional rates come with no posterior. No whole-race fact appears in any tool output.

## Decisions

The agent submits one decision at a snapshot, not a whole plan: PIT with a compound, or
STAY_OUT with `reassess_after_laps`. A policy returns it as `Decision(None,
submission=...)`, and `eval.decisions.to_plan(snapshot, decision, belief, max_stops)`
turns it into the plan the world scores:

- PIT c is `{"action": "pit", "compound": c}`, a stop at the end of this lap;
- STAY_OUT with `reassess_after_laps` n is `{"action": "stay", "laps": n}`;
- the continuation from the snapshot after the action is
  `optimizer_bridge.optimize_batch(advanced, belief, stops left)`, on the belief's fit at
  the decision snapshot, so no lap after the decision enters it;
- `max_stops` counts future stops from the decision snapshot, as the reference optimizer
  does: after a PIT the continuation gets `max_stops - 1` (at least 0), after a STAY_OUT
  `max_stops`. Every plan from a decision lies inside the space the reference searched, so
  an agent row with negative regret (beyond 1e-6 s) is a bug, and `summarize` warns
  (`AgentRegretWarning`). Baselines keep their exception: an actual three-stop can beat a
  two-stop reference. No tyre-life limit binds any plan (R8 is retired, docs/rules.md): an
  over-long stint is priced by the wear cliff, the world's from the whole race and the
  belief's from what is known at the snapshot ("The wear cliff" below), and a plan is invalid
  only for a published-regulation violation;

- `Simulator.expand_action` joins the two. A stay of n at least the laps left runs the
  current set to the flag.

A submission that cannot become a legal plan (no compound, a compound the belief does not
model, n below 1, no legal continuation) is an invalid row.

## The belief_optimizer policy

`--policies baselines belief_optimizer agent` adds a baseline with no model call: at each
snapshot it takes the plan `optimizer_bridge.optimize_batch` picks on the belief simulator (the
one the agent's tools use, fitted on practice plus race laps up to the snapshot), with no
decision imposed, and the world scores that plan like any other. It answers what the language
model adds over running the optimizer on what it can see. It is planned by
`eval.decisions.belief_optimum`, with the same belief fit, seed (seed + `BELIEF_SEED_OFFSET`),
samples and future-stop limit as the agent continuation in `to_plan`. It follows every rule
(R1-R7); a plan the world rejects is an `invalid` row with the reason. Rows carry the usual
plan, stops, first pit, regret and the (retired) R8 stamps, blank; the agent-only columns (`decision`, `p_pit`,
tool calls, tokens) are blank. It is not in the `baselines` group, so default commands and
existing result files are unchanged; select it by name.

## Rule baselines: run_to_flag and one_stop_total

Two deterministic rule policies (`src/eval/policies.py`), selected by name
(`--policies run_to_flag one_stop_total`) and, like `belief_optimizer`, not in the `baselines`
group, so default commands and existing result files are unchanged. They were added because
`fixed_one_stop` always makes exactly one more stop from the snapshot, also after the driver
has made their stop, when the sensible plan is usually to run to the flag.

**run_to_flag.** Make no further stop when the rules allow it; otherwise the fewest stops
that make the plan legal, each as late as the rules allow:

1. The plan is the current set to the flag. If it passes every rule (R1-R4 and the
   belief's known compounds, checked with `belief.validate` on the plan as a contract, sets
   assigned by `to_contract` as for every baseline), that is the plan.
2. Otherwise, for k = 1, 2, ... up to `max_stops` (the future-stop limit every policy gets,
   2 in the committed runs), take the k-stop plans that pass every rule and submit the first
   one in this order: the latest first stop, then the latest second stop (pit laps compared
   first stop first), then the compounds of the new stints in the preference order SOFT,
   MEDIUM, HARD, then any other modelled compound alphabetically, compared stint by stint.
   The new stints may use any compound the belief models, the current one included (a new set).
   The first k with a legal plan wins.
3. If no plan with at most `max_stops` stops is legal, the plan is the current set to the
   flag; the world rejects it and the row is `invalid` with the world's reason.

The compound rule is fixed: since the forced stops are placed as late as the rules allow, the
stint after one is as short as they allow, and the softest legal compound is the usual choice
for a short final stint. Placing the stops first and choosing the compound second is part of
the rule: when a later stop needs a harder compound, the later stop wins. With R8 retired the
forced stops come from the published rules (R2 minimum stint, R3 two compounds) and the belief's
as-of tyre-life estimate at the snapshot, the information every policy has ("The wear cliff",
rule baselines).

**one_stop_total.** A one-stop race in the ordinary sense, one stop in the whole race:

- If the driver has not stopped yet in this race (the snapshot's completed-stint history,
  `snapshot.stints`, is empty), the decision is `fixed_one_stop`'s, through the same eval
  wrapper: the stop at the middle of the remaining laps (moved to respect the minimum stint,
  R2), onto the first of HARD, MEDIUM, SOFT the belief models that differs from the current
  compound (`_one_stop_within_limit` moved it earlier under R8; it runs only under the legacy
  setting). Where `fixed_one_stop` is unavailable or invalid, so is this row.
- If the driver has stopped at least once (`snapshot.stints` is not empty), it is
  `run_to_flag`.

What they read. Both are functions of the snapshot and the published rules only: the laps
completed and left, the current compound, its age and laps on the set, the stint history, the
tyre sets, the set of compounds the belief models at the snapshot (fitted only on what is known
there), the minimum stint (R2) and the future-stop limit. Neither reads a lap time, a pace or
wear fit, the belief optimizer, the reference or any tuned constant, and neither reads the pit
loss. Under R8 they also knew the race-wide stint limits, a hindsight race constant; R8 is
retired, and their tyre life is the belief's as-of estimate, so they know nothing a driver could
not know at the snapshot ("Information contract").
A plan the world rejects is an `invalid` row with its reason, as for any policy.

## Information contract

What each side may know, and the test that holds it (`tests/test_tyre_life_future_blindness.py`).

**A policy** at (race, lap L), and everything it calls (the belief simulator, the belief
optimizer, the continuation of a decision, the baselines, the agent's tools), may use only:
this weekend's practice, sprint and qualifying; race laps up to L (every car's); the published
regulations (R1-R4, the race distance, the future-stop limit); and priors fixed in config or
estimated from races strictly before this one (`race_ids.as_of`). No rule a policy sees depends
on a lap after L, and the agent's tools state no whole-race fact. The test truncates or
scrambles every lap after L (synthetic race at three snapshots, and 2021 / 2023 Spain at laps
10, 25 and 40) and asserts that the belief's cliff ages, every tool output and error string,
the belief optimizer's plan, the continuation of a PIT and a STAY_OUT, and every baseline plan
are unchanged; with the legacy R8 setting the same check fails, which is the leak it guards
against. `actual_strategy` is exempt by definition: it replays what the team did.

**The scorer** (the world simulator and the hindsight reference) may use the whole race. It
expresses tyre life as lap time (the wear cliff below), never as a rule handed to policies.

The cross-race tables a policy reads (the proxy's steps, weights and sd floor among them) are
read as of the race since the cross-race change: "Cross-race tables, as of the race" under the
first "Information contract" section above. The other known leaks are listed in
`tests/future_blindness.py` (KNOWN_LEAKS) and handled by their own changes.

## The wear cliff

A degradation model's wear is a line (or a line with a hinge), so with nothing else an optimizer
plans absurd stints (a 59-lap SOFT at 2024 Emilia Romagna, 38-lap SOFT at 2023 Spain). R8 forbade
them with a hindsight rule; the cliff prices them. Inside a compound's support the fitted model
holds; past its support edge E a lap costs

    min(kappa x (age - E)^2, cap)   seconds more,   kappa = 0.05 s/lap per lap^2, cap = 4 s/lap

(`src/sim/cliff.py`, `SimConfig.wear_cliff`, added in `Simulator.simulate_batch` to green laps at
their effective age, before the track factor). A plan 2 laps past the edge pays about 0.3 s; a
dead tyre pays at most 4 s a lap, so one absurd plan cannot dominate a mean. Nothing is invalid
for tyre age. The data cannot locate a cliff by itself: fitted curvature is significantly negative
on several race/compound pairs (2023 Spain HARD and MEDIUM, 2024 Emilia Romagna HARD and MEDIUM,
2025 Qatar HARD: long runs are survivorship-selected), so past the support the model's knowledge
ends and this stated extrapolation takes over. kappa anywhere in 0.02-0.10 gives the same lap-0
stop counts on the development races, so it is 0.05 and not tuned (design study on the six
development races; the capped form is the study's recommendation, adopted).

**The edge, judge and predictive simulator.** Both start from the same historical prior on tyre
life (`cliff.history_prior`: races strictly before this one by date, held-out races never), so they
agree as far as they can, and each extends it outward only where it has lap data.

- *Predictive simulator (belief, every window but `full`)*: per sample, max(the oldest tyre age
  seen so far, prior + sd z), the ages seen being this weekend's practice and sprint and race laps
  up to the fit's calibration point (every car; never an advanced snapshot's lap, since a plan's
  continuation is its hypothetical future), z standard normal from its own stream
  (`engine.CLIFF_STREAM`, so every other draw is unchanged). Practice ages are run lengths, not
  tyre life (2023 Spain HARD: 7 in practice, 50 in the race), so they are only a lower bound.
- *Judge (world, `full`)*, `wear_cliff.world_edge`, three sources with the same cost form:
  (a) `history_only`, the prior's point edge, nothing from this race; (b) `history` (the default
  until the fidelity scorecard chooses between a and b by judge-versus-reality accuracy out of
  sample), the prior extended by the ages this race has lap data for, max(prior, oldest age run),
  the judge's one whole-race input here; (c) `race`, the oldest age run in this race (for an
  unraced compound the proxy's practice-age rule), the earlier design. The world's
  `fit_report["support_edge"]` (and `Simulator.cliff_ledger`) records per compound the edge, its
  source ("history:<tier>", "history:<tier> extended by race data to age N", "race", ...) and
  whether it used this race's laps, for the fidelity tool's hindsight ledger.

**The prior.** `wear_cliff.prior_source: tyre_history` (the default) is the shared history model,
`src/sim/tyre_history.py` `tyre_life_prior(race_id, compound)`: C-number and circuit weighted,
strictly as-of by date, held-out races never, its own fallback chain (tiers 1-4) down to a
documented constant; its validation reports a typical miss of 8 laps (3 with same-circuit
history), 68/90 percent intervals covering 65/90 percent, and a slight short bias, which the
belief's rule that observed ages extend the edge offsets. A race id with no date (a synthetic
race) takes the stub's constant tier. `prior_source: stub` is the earlier as-of stub,
`cliff.life_prior`, an explicit fallback chain, each tier with a wider sd (a share of race
distance): the same circuit in an earlier season, scaled by race distance
(`circuit_history:<race>`, sd 0.10); else the median share of race distance the compound's oldest
stint ran in earlier races at any circuit (`history_fraction`, sd 0.15; 2025 Qatar left out, its
stints were capped by regulation); else a neighbour's prior times the as-of life ratio (the median
of the per-race ratios of earlier races in `compound_steps.json`, sd 0.18); else a documented
constant, `prior_life_fraction` of race distance (SOFT 0.4, MEDIUM 0.55, HARD 0.7, a priori, sd
0.20, the widest). Every dry compound gets a prior whatever history exists, and the belief never
falls back to a quantity of this race. Its history is the committed table
`configs/sim/tyre_life_history.json` (written by `python -m src.sim.cliff history --races ...`
from the calibration races) plus, at run time, every cached race strictly before the race
(`wear_cliff.scan_cache`; held-out races are left out before anything is loaded), so newly cached
seasons enter without a code change. The measurements below (the prior's accuracy, the lap-0
tables, the cross-race run) were taken with the stub prior, before the switch to
`tyre_history`; the integration step re-measures them.

The prior's own accuracy, scored against the oldest age actually run (the hindsight fact is the
measurement target here, not an input), stub prior on the calibration races:

| race | compound | history edge | sd | tier | oldest run | run - edge | within 1 sd |
|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | SOFT | 20.8 | 10.4 | config_fraction | 23.0 | 2.2 | True |
| 2019_10_united_kingdom | MEDIUM | 28.6 | 10.4 | config_fraction | 39.0 | 10.4 | True |
| 2019_10_united_kingdom | HARD | 36.4 | 10.4 | config_fraction | 40.0 | 3.6 | True |
| 2021_04_spain | SOFT | 29.2 | 9.9 | history_fraction | 32.0 | 2.8 | True |
| 2021_04_spain | MEDIUM | 49.5 | 9.9 | history_fraction | 42.0 | -7.5 | True |
| 2021_04_spain | HARD | 50.8 | 9.9 | history_fraction | - | - | - |
| 2022_02_saudi_arabia | SOFT | 23.2 | 7.5 | history_fraction | - | - | - |
| 2022_02_saudi_arabia | MEDIUM | 34.7 | 7.5 | history_fraction | 16.0 | -18.7 | False |
| 2022_02_saudi_arabia | HARD | 38.5 | 7.5 | history_fraction | 41.0 | 2.5 | True |
| 2023_01_bahrain | SOFT | 26.4 | 8.5 | history_fraction | 25.0 | -1.4 | True |
| 2023_01_bahrain | MEDIUM | 36.3 | 8.5 | history_fraction | 10.0 | -26.3 | False |
| 2023_01_bahrain | HARD | 45.3 | 8.5 | history_fraction | 29.0 | -16.3 | False |
| 2023_07_spain | SOFT | 32.0 | 6.6 | circuit_history:2021_04_spain | 28.0 | -4.0 | True |
| 2023_07_spain | MEDIUM | 42.0 | 6.6 | circuit_history:2021_04_spain | 32.0 | -10.0 | False |
| 2023_07_spain | HARD | 50.8 | 9.9 | history_fraction | 50.0 | -0.8 | True |
| 2024_07_emilia_romagna | SOFT | 27.7 | 9.4 | history_fraction | 8.0 | -19.7 | False |
| 2024_07_emilia_romagna | MEDIUM | 30.5 | 9.4 | history_fraction | 38.0 | 7.5 | True |
| 2024_07_emilia_romagna | HARD | 48.1 | 9.4 | history_fraction | 54.0 | 5.9 | True |
| 2025_23_qatar | SOFT | 24.2 | 8.5 | history_fraction | 25.0 | 0.8 | True |
| 2025_23_qatar | MEDIUM | 32.5 | 8.5 | history_fraction | 25.0 | -7.5 | True |
| 2025_23_qatar | HARD | 43.8 | 8.5 | history_fraction | 25.0 | -18.8 | False |

**The agent's tools** report the belief's edge per compound (`model_report`'s `tyre_life`: the
longest run seen so far and where, the prior and its sd and tier, the cost form, and the sentence
"longest run seen so far on this compound: N laps (...); pre-race tyre-life estimate M laps (sd
S, tier); wear beyond that is uncertain"); it is never an error or a limit.

**Rule baselines** (`fixed_one_stop`, `greedy_crossover`, `run_to_flag`, `one_stop_total`) get the
same as-of estimate every policy has: their limit is the belief's point edge per compound at the
snapshot, margin 0 (`eval.policies._stint_limit`). Without it run_to_flag would run any set to the
flag. Their plans are still invalid only for a published-regulation violation.

**Switches** (each in `SimConfig`, hashed): `wear_cliff.enabled: false` removes the cost (the
legacy R8 setting does); `wear_cliff.world_edge`; `wear_cliff.prior_source`;
`min_race_rows_to_fit: N` leaves a compound with fewer than N clean race rows to the compound
proxy instead of fitting it from a handful of laps (default off; at 50 on the calibration races the
lap-0 references are unchanged except 2024 Emilia Romagna's SOFT, 13 to 12 laps under edge (c)).

Over the 19 raced (race, compound) pairs the prior is 5.0 laps long on average (mean absolute
error 8.8 laps) and the oldest age run lies within one prior sd in 68 percent of them. Its large
misses are compounds that were barely raced, whose oldest age says little about tyre life (2023
Bahrain MEDIUM 10, 2022 Saudi Arabia MEDIUM 16, 2024 Emilia Romagna SOFT 8), and 2025 Qatar, whose
stints were capped at 25 laps by regulation.

### The world's wear fit, and the measures before and after

Selected by the cross-race rule (`python -m src.sim.calibrate cross run --races 2021_04_spain
2023_07_spain 2023_01_bahrain 2019_10_united_kingdom 2022_02_saudi_arabia 2024_07_emilia_romagna
2025_23_qatar --no-tau-grid`, explicit calibration races; held-out races never loaded). The rule
was amended before this run: `MIN_RACES` 8 could never pass on the seven calibration races left
after the hold-out, so it is 5, and a candidate must also be worse than its comparator on at most
one race (`cross_race.MAX_WORSE_RACES`, on G for the world, on stop-gain for the belief). With the
cliff in place, `driver_effects_hinge` (linear_fuel_hinge, knot 25, driver offsets on the race
rows) passes the world rule against the SimConfig default: G mae_offset 17.89 to 17.12 s, paired
difference -0.77 [-1.16, -0.38], worse on no race; stop-gain bias -0.18 to -0.12 s, mae 0.53 to
0.44. It is the eval world (`config.WORLD_DEGRADATION`, `score.race_simulators`,
`configs/sim/world.yaml`); the SimConfig default stays linear_fuel, the comparator the arms are
overlays on. Per-compound knots at 75 percent of the longest life (`knot_life_fraction`) and
dropping negative hinges (`hinge_lower_bound: 0`) were measured and did not pass
(`driver_effects_hinge_life`: G CI upper end +0.01; `driver_effects_hinge_bound` against
`driver_effects_hinge`: +0.06 [-0.23, 0.41]); both stay as options. A negative fitted hinge acts
only inside the support (2023 Spain HARD -0.081 s/lap past 25): past the edge the cliff applies
whatever the hinge does. The belief: no arm passes its rule, so `wear_nonneg` stays; driver effects
alone (`driver_effects_nonneg`) improve G (-2.45 [-4.62, -0.21]) but not stop-gain (-0.02 [-0.06,
0.02]).

Per calibration race, before (main: linear_fuel world with R8; belief wear_nonneg) and after (the
`driver_effects_hinge` world with the cliff, edge source (c) in the stop-count column; the belief
with the as-of cliff). G is the race's mae_offset at horizons 10-49 (s), SG the stop-gain bias and
mae at real stops (s, predicted less real gain); stops are the lap-0 reference's against the
field's mode.

| race | world G before / after | world SG bias before / after | world SG mae before / after | belief G before / after | belief SG bias / mae (both) | field mode | lap-0 ref stops before / after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 United Kingdom | 16.84 / 16.51 | -0.04 / +0.07 | 0.34 / 0.28 | 27.79 / 27.63 | -0.81 / 1.16 | 1 | 1:17 / 1:17 |
| 2021 Spain | 10.93 / 9.71 | -0.34 / -0.16 | 0.53 / 0.38 | 19.12 / 19.22 | -1.01 / 1.09 | 2 | 1:12 2:7 / 1:2 2:17 |
| 2022 Saudi Arabia | 44.14 / 42.62 | +0.31 / +0.01 | 0.55 / 0.37 | 28.49 / 28.66 | -0.22 / 0.60 | 1 | 1:14 / 1:14 |
| 2023 Bahrain | 20.09 / 19.94 | -0.44 / -0.13 | 0.77 / 0.73 | 34.33 / 34.34 | -0.15 / 0.85 | 2 | 1:1 2:16 / 2:17 |
| 2023 Spain | 8.07 / 7.75 | -0.53 / -0.31 | 0.66 / 0.54 | 11.66 / 11.94 | +0.02 / 0.67 | 2 | 1:20 / 1:20 |
| 2024 Emilia Romagna | 13.27 / 11.80 | -0.13 / -0.18 | 0.37 / 0.31 | 17.03 / 19.09 | +0.36 / 0.59 | 1 | 1:17 2:2 / 1:17 2:2 |
| 2025 Qatar | 11.93 / 11.53 | -0.11 / -0.12 | 0.50 / 0.48 | 29.57 / 29.68 | -0.78 / 0.89 | 2 | 1:5 2:13 / 1:18 |

Lap-0 stop-count mode matches the field's in 5 of 7 races before (R8) and after. 2023 Spain stays
one-stop under every world model (its HARD and MEDIUM show negative curvature); 2025 Qatar now
one-stops because its mandated 25-lap maximum is not modelled (docs/rules.md R8, "Not modelled").

Longest planned stint (end age, lap-0 references of every field car) against the longest run,
before (R8) and after under each judge edge source (world `driver_effects_hinge`):

| race | longest run S / M / H | before, R8 | after, (c) race | after, (b) history (default) |
| --- | --- | --- | --- | --- |
| 2021 Spain | 32 / 42 / - | S35 M33 H21 | S33 M31 | S29 M27 H41 (proxied HARD; one-stop) |
| 2023 Spain | 28 / 32 / 50 | S31 M35 H38 | S22 M18 H49 | S22 M18 H49 |
| 2023 Bahrain | 25 / 10 / 29 | S27 H28 | S19 H21 | S19 H21 |
| 2019 United Kingdom | 23 / 39 / 40 | S26 M31 | S26 M31 H33 | S26 M31 H33 |
| 2022 Saudi Arabia | - / 16 / 41 | M18 H34 | M11 H41 | M11 H41 |
| 2024 Emilia Romagna | 8 / 38 / 54 | S11 M29 H35 | S13 M31 H33 | S33 M31 H30 |
| 2025 Qatar | 25 / 25 / 25 | S6 M28 H28 | S27 M28 H30 | S18 M23 H39 |

No plan is absurd under (c): the worst overshoot is 5 laps (Qatar HARD, the unmodelled cap). Under
(b) the edge sits where history puts it when this race ran a compound less: 2024 Emilia Romagna's
SOFT (8 laps raced, a slightly negative fitted wear from 11 race laps) is planned to 33, and 2021
Spain turns one-stop on the proxied HARD (lap-0 stop mode 4 of 7). That is the trade the fidelity
scorecard measures; (b) is the default until it does.

### Fuel identification (measured, not adopted)

The fuel slope's stated interval was far too tight: the config prior (-0.045 s/lap, tau 0.008)
covers the hindsight fuel slope within one sd in 1 of 7 calibration races (2 of 7 within 1.645
sd; rms error 0.030). The physics prior (`fuel_practice_longrun`: -0.03 s/kg x 110 kg / race laps
plus practice evolution, no race lap and no other race read) with its sd set by coverage, 0.03
s/lap (the 68th percentile of |prior - hindsight| over the seven races), covers 5 of 7 (6 of 7 at
1.645 sd). The posterior's coverage barely moves (fuel 68/90 percent intervals cover 26/44 percent
of (race, lap) points before, 33/51 percent with the physics prior; wear 39/57 percent either way),
because the posterior sd is set by the data term, which treats residuals as independent while their
lag-1 autocorrelation is 0.78-0.94: that is the noise model, a separate change. The joint draws
carry the fuel-wear correlation (median draw correlation -0.22 against -0.15 in the covariance; the
gap is the wear bound's truncation). In the cross-race run, `fuel_physics_driver_effects` (the
physics prior, tau 0.03, driver effects) improves the belief's stop-gain mae 0.84 to 0.74 (paired
-0.10 [-0.15, -0.04], worse on one race) and its bias -0.37 to -0.19 s, and G mae 24.4 to 21.7, but
the G difference's CI upper end, +2.20 s, is over the rule's +2 s non-inferiority margin, so the
rule does not adopt it and belief.yaml is unchanged. The arm files
(`configs/sim/arms/fuel_physics*.yaml`) and the calibrator's `sd_s_per_lap` stay for the follow-up.

### Fuel anchoring

Within a stint tyre age and race lap rise together, so the fuel slope and the wear slopes are
confounded, most of all before the first stops. The learned ridge calibrator fixed most of the
belief's fuel error through the fuel factor alone, but it learns from later races. Fuel anchoring
gets that gain with no hindsight.

**The prior** (`fuel_prior.anchored_prediction`, `race_history.as_of_fuel(..., mode="anchored")`).
The physics slope is -0.03 s/kg x (practice lap / 90 s) x 110 kg / scheduled laps. The time a
kilogram costs is taken as a fixed share of the lap, so the 0.3 s per 10 kg rule of thumb holds at
a typical 90 s lap. The practice lap is this weekend's median clean practice lap, which is pre-race.
The as-of history supplies the ratio target / physics of the races before this one. The target is
each race's free-air full-window fuel slope in `configs/sim/race_history.csv`, which now also
records `practice_lap_s`. That ratio has an a priori N(1, 0.2^2) prior and is combined with the
ratios' mean by precision. The predictive sd is the ratios' spread, or 0.2 with fewer than three
earlier races, plus the uncertainty of their mean, times the physics slope. With no earlier race the
prior is the physics slope with a 20 % sd. Every constant is a priori and listed in
`provenance.POLICY_SIDE["fuel_physics"]`.

Forward-chained over the 19 calibration races, each race priced only by the races before it,
against the hindsight target:

| prior | mae s/lap | bias | 68% / 90% coverage |
|---|---|---|---|
| circuit history (#102, the previous default) | 0.0157 | -0.0046 | 0.53 / 0.79 |
| lap-scaled physics, a priori | 0.0074 | +0.0031 | 0.68 / 0.89 |
| anchored (physics x as-of ratio, adopted) | 0.0084 | -0.0009 | 0.63 / 0.89 |

The unscaled physics slope has mae 0.0099. Scaling by the lap time takes it to 0.0074, and the
ratio's spread falls from 0.21 to 0.17.

**Both simulators.**

- **Belief.** `fuel_anchor.prior: anchored` (read by the as_of_history calibrator) makes the anchored prior the to_snapshot fit's fuel
  prior, with its sd as the fuel tau. `fuel_anchor.practice: anchored` fixes the practice fit's
  in-stint fuel column at the same slope instead of -0.045, so the practice wear that the race fit
  shrinks toward is net of the same fuel. Both are now on in belief.yaml. The posterior covariance,
  and hence the draws, carries the fuel-wear correlation as before.
- **Judge.** `fuel_anchor.full_fit: true` shrinks the full fit's fuel slope toward the a priori
  physics slope, using the race's laps and practice lap and no other race. The penalty is residual
  variance / sd^2, so the sd is a prior sd in s/lap. It is off by default (below).

**Measured** with the fidelity scorecard (`python -m src.sim.fidelity run`, PR #93) on the 18
calibration races (the six development races and twelve 2021-22 weekends). Settings: 6 drivers per
race, a snapshot every 5 laps, 20 draws. The belief is compared with the default judge.

| measure | before | belief anchored | belief and judge anchored |
|---|---|---|---|
| plan ranking, Spearman | 0.848 | 0.849 | 0.849 |
| judge-regret of the belief's favourite, s | 5.47 | 5.29 | 5.32 |
| pit-now agreement | 0.756 | 0.762 | 0.756 |
| replay of real strategies: close-pair order | 0.637 | 0.640 | 0.640 |
| replay: relative gap error, s | 21.74 | 21.72 | 21.72 |
| replay against reality (G): bias, s | -3.5 | +8.8 | +8.8 |
| G pace-offset mae / bias, s | 34.6 / +7.2 | 33.9 / +15.3 | 33.9 / +15.3 |
| fuel against the judge: mae, s/lap | 0.017 | 0.010 | 0.010 |
| fuel 68% / 90% coverage (component, k 0/10/20/mid) | 0.40 / 0.63 | 0.49 / 0.72 | 0.49 / 0.76 |
| fuel 68% / 90% coverage (every B snapshot) | 0.37 / 0.55 | 0.39 / 0.60 | 0.39 / 0.60 |
| wear 68% / 90% coverage (every B snapshot) | 0.44 / 0.60 | 0.42 / 0.57 | 0.42 / 0.57 |
| remaining time 68% / 90% coverage | 0.55 / 0.72 | 0.49 / 0.64 | 0.49 / 0.64 |
| plan-versus-plan gap 68% / 90% coverage | 0.50 / 0.68 | 0.51 / 0.69 | 0.51 / 0.69 |
| belief stop-gain bias, s | +0.07 | +0.10 | +0.10 |

Against the judge with driver effects (`degradation.params.driver_effects: race`) the pattern is the
same. Spearman goes 0.834 to 0.838, regret 5.35 to 5.20 s, pit-now 0.746 to 0.753, fuel coverage
0.40 / 0.63 to 0.43 / 0.69, and the order measures are unchanged.

**Decision.** The belief's anchoring is adopted (belief.yaml). It improves plan-ranking agreement
and regret and fuel accuracy and coverage, and it leaves the order measures as they were.

The replay of real strategies moves from 3.5 s fast to 8.8 s slow. Its pace-offset mae improves
(34.6 to 33.9 s). A uniform offset does not change a decision, and the remaining-time interval is
widened in "Honest uncertainty" below. Anchoring the judge too leaves that offset as it is (the
belief's bias against the judge with no SC is +10.6 s, +10.4 s with both anchored), so the offset
is the belief's against reality and not a mismatch of anchors.

The judge's anchor stays off by default. The whole-race data dominates its prior: the judge's fuel
moves by about 0.001 s/lap. Its order measures barely move: close-pair 0.693 to 0.701 (about one
pair), relative gap 15.58 to 15.57 s, judge stop-gain bias -0.250 to -0.256 s. That is within
noise, so the switch is kept for the rerun to decide.

Per-race fuel and wear before and after are in the PR description.

## Proxied compounds

A strategy is not banned because nobody ran it. The world's full-window fit models a compound
only when it has `min_samples` (5) clean race laps (`degradation._fuel_fit` keeps `_usable`
compounds; the rest go to `fit_report["unmodelled"]` with the lap count), and it never reads
practice: the practice prior enters only the belief's `to_snapshot` and `pre_race` fits (the
shrinkage prior of `_shrunk`, whose compounds are the prior's plus the raced ones). So at
2022 Saudi Arabia, where SOFT was run in practice but never raced, the belief priced SOFT and
the world did not: every plan with a later SOFT stint was "Unknown compound 'SOFT'" (20 of the
21 invalid agent rows of dev6 pass 1). With `SimConfig.compound_proxy.enabled` (the default;
`false` restores the simulator as it was, bit for bit) every fit prices HARD, MEDIUM and SOFT
(`src/sim/proxy.py`).

**The rule.** A compound with `min_samples` clean race rows in that simulator keeps its fit
(the belief: rows up to its calibration point). For any other, pace (the fresh base) and wear
are each a precision-weighted blend, from the nearest raced compound (the anchor), of every
source that exists for it:

| source | pace | wear | exists when |
| --- | --- | --- | --- |
| practice long runs this weekend (FP sessions, plus the Sprint on a sprint weekend) | anchor + (practice base_c - practice base_anchor) | anchor + (practice wear_c - practice wear_anchor) | the compound and the anchor have practice long runs |
| qualifying this weekend | anchor + the qualifying gap (below) | none | the compound and the anchor have qualifying laps |
| the cross-race neighbour step (`configs/sim/compound_steps.json`) | anchor x (1 + pace_frac) per step | anchor x wear_ratio per step | always |

value = sum_i w_i v_i / sum_i w_i, with fixed weights, one set for all races, w = 1 / that
source's mean squared error alone in the hidden-compound validation below (all calibration
races). Pace and wear have separate weights: qualifying informs pace only, and a stint error
mixes the two. The pace weights use each source's fresh-pace error, the wear weights its wear
error:

| | practice | qualifying | cross-race step |
| --- | --- | --- | --- |
| pace: fresh-pace MSE, s^2 (n) | 2.671 (16) | 1.091 (6) | 0.247 (19) |
| pace weight (share when all three exist) | 0.374 (7%) | 0.917 (17%) | 4.055 (76%) |
| wear: wear MSE, (s/lap per lap)^2 (n) | 0.0139 (16) | | 0.0058 (19) |
| wear weight (share) | 71.8 (29%) | | 172.7 (71%) |
| for comparison, 15-lap stint MSE, s^2 (n) | 322.4 (16) | | 31.8 (19) |

(Weights from the 15-lap stint error would give practice 9% against the step's 91%; they
cannot weigh qualifying, which has no wear.) The blended value's prior sd is the larger of the
blend's own leave-one-race-out error (0.585 s pace, 0.079 s/lap per lap wear,
`compound_proxy.sd_floor`) and the cross-race step's spread times the anchor value. Tier 4
(the same circuit in other seasons) is not built: `proxy._tier4_pace` and `_tier4_wear` are
the extension point. A belief compound with fewer than 5 race rows by its calibration point is
re-priced the same way, relative to a raced compound: **this changes what the agent sees on
any race where a compound has under 5 race rows at the snapshot** (before, it stayed at
practice plus the config base delta). A fit with no raced compound yet (the belief before its
first race laps, `pre_race`) keeps its practice compounds and anchors the missing ones on them.
Weights, steps and sd floor are read through the config, so `config_hash` covers them.

**The qualifying correction.** A qualifying lap is a low-fuel push lap on an evolving track, so
it is never compared with a race lap directly. Each driver's best timed lap per (segment
Q1/Q2/Q3, compound) (`data.qualifying_frame`: a lap time, no pit in or out, a dry compound,
track status green) is one row of a least-squares fit lap = driver + segment + compound, slow
rows trimmed against the fit as `data.robust_filter` trims practice (2.5 robust sd, two
passes; fast rows never cut). Only the compound gap q_c - q_anchor is used, added to the
anchor's race base. Fuel cancels in it (every qualifying lap carries little fuel, and the race
base the gap is added to carries the race's fuel level, so the race fit's fuel slope is not
needed); car and driver pace cancel (the driver term); evolution and engine modes between
segments cancel (the segment term, which takes the pre-2020 Q3 modes too). Not corrected:
evolution within a segment, and push against managed driving. A compound needs 2 drivers who
also set a row on another compound. Without qualifying laps on the anchor, qualifying does not
enter. HARD is never run in qualifying, so it only ever informs SOFT against MEDIUM, and at
2022 Saudi Arabia it informs nothing (2 drivers on MEDIUM, one trimmed). The measured gap is
shown to the agent for every compound that has one (`qualifying_gap_s` in the model report,
e.g. 2021 Spain "SOFT: ...; qualifying gap to MEDIUM -0.94 s on fresh tyres (3 drivers)"), raced
or not, since fresh-tyre pace is the datapoint the race fit cannot give.

**The step table** (`python -m src.sim.compound_steps estimate`; calibration races only, the
four held-out races never loaded). Per race, the full-window world fit with the proxy off; a
pair counts when both compounds have at least 50 clean race rows (`MIN_PAIR_ROWS`); the
value is the median over races, sd the sample sd across them. Before and after that filter:

| pair | key | all raced pairs (before) | at least 50 rows (shipped) |
| --- | --- | --- | --- |
| MEDIUM-HARD | pace_frac | -0.0035, sd 0.0038, n 6 | -0.0027, sd 0.0033, n 5 |
| MEDIUM-HARD | wear_ratio | 1.53, sd 0.85, n 6 | 1.40, sd 0.65, n 5 |
| MEDIUM-HARD | life_ratio | 0.64, sd 0.26, n 5 | 0.67, sd 0.24, n 4 |
| SOFT-MEDIUM | pace_frac | +0.0019, sd 0.0061, n 6 | +0.0001, sd 0.0026, n 3 |
| SOFT-MEDIUM | wear_ratio | 0.60, sd 0.88, n 6 | 0.72, sd 0.33, n 3 |
| SOFT-MEDIUM | life_ratio | 0.76, sd 0.88, n 5 | 0.76, sd 0.14, n 3 |

The filter drops 2023 Bahrain (MEDIUM, 8 rows) from both pairs and 2024 Emilia Romagna (SOFT,
11 rows) and 2025 Qatar (SOFT, 34 rows) from SOFT-MEDIUM, which rests on 2019 United Kingdom,
2021 Spain and 2023 Spain only. **Its pace step is +0.0001, zero to within 0.01 s on a 90 s
lap: on these race fits a fresh SOFT is no faster than a fresh MEDIUM, and its wear ratio
(0.72) says SOFT wears less.** After the ordering constraint a step-priced SOFT is a copy of
MEDIUM. The yardstick matters here: the step reflects the simulator's own race fits, which
under-separate SOFT from MEDIUM (the pooled fit confounds compound with when and by whom it was
run). A proxied compound has to sit where the world's own fits would put it, or the search
exploits the difference; the table is re-estimated when driver effects are adopted.

**Ordering, R8, uncertainty, reporting.** A softer compound is never slower fresh and never
wears less than a harder one: the constraint moves proxied values only, never a raced one
(where raced values are out of order the proxied one is put between them, "conflict"); a
proxied wear is also floored at 0 and at the belief's wear bound. A compound no car raced gets
an R8 limit: its oldest practice tyre age, else the nearest raced compound's limit times the
life ratio, plus the margin (docs/rules.md R8; 2021 Spain HARD: 18 in FP1, longest practice run
13 laps, cap 21; 2022 Saudi Arabia SOFT: 18 in FP1, longest run 12 laps, cap 21). A blended
parameter is linear in its anchors, so `fit_report["cov"]` carries it as T cov T' + tau^2 and
posterior draws move it with them. `fit_report["proxy"]` holds, per compound, `pace_sources` and
`wear_sources` ([{source, weight, anchor, value, step}]), `pace_source` / `wear_source` (the
sources joined by "+", or "race"), the steps, the sds, the ordering notes and
`qualifying_gap_s`. The model report flags `pace_from_<sources>:<compound>`; the agent's model
report and `estimate_tyre_degradation` carry the sources, their weights, the qualifying gap and
`price_basis` ("SOFT: pace from practice long runs 8%, the cross-race step 92% (anchor MEDIUM);
wear from practice long runs 29%, the cross-race step 71% (anchor MEDIUM)" at Saudi). Score rows
carry `proxied_compounds` and `opt_proxied_compounds` (the plan's and the reference's compounds
the world prices by proxy, ";"-joined, empty for none) when the world's proxy is on. **Results
can be read with and without the rows where either is non-empty.**
`calibrators.hindsight_targets` and `cross_race.wear_check` leave proxied compounds out.

### Validation

`python -m src.sim.compound_steps validate` (calibration races only; outputs under
`reports/calibration/compound_proxy/`; it writes the weights and the sd floor into the table).
For each calibration race and each compound that was raced, its race laps are hidden (its rows
dropped from the race frame), the world refitted, and the compound priced again, then compared
with the real race fit: the fresh pace at mid-race, the wear, and the time of a 15- and a
25-lap stint on a new set centred on mid-race. Each source alone uses the step table without
that race; **the blend is leave-one-race-out throughout: for each race its weights come from the
other races' source errors and its step table from the other races** (the shipped constants
use every race). Proxy minus truth, mean signed / mean absolute error:

| priced by | n | fresh pace, s | wear, s/lap per lap | 15-lap stint, s | 25-lap stint, s |
| --- | --- | --- | --- | --- | --- |
| practice alone | 16 | +0.18 / 1.06 | +0.020 / 0.067 | +5.1 / 10.0 | +10.9 / 17.7 |
| qualifying alone (pace) | 6 | -0.15 / 0.85 | | | |
| cross-race step alone | 19 | +0.01 / 0.38 | +0.004 / 0.043 | +0.6 / 4.4 | +1.6 / 10.4 |
| **blend (leave-one-race-out)** | 19 | +0.02 / 0.47 | +0.007 / 0.048 | +1.1 / 4.9 | +2.6 / 10.6 |
| (the old best-single-tier chain, old step table) | 19 | +0.06 / 0.99 | +0.019 / 0.064 | +3.3 / 9.6 | +7.8 / 16.4 |

The blend's root mean squared errors (0.585 s, 0.079 s/lap per lap) are the sd floor. The
step alone is slightly better than the blend on every column: practice and qualifying, even
at 7% and 17% of the pace weight, add more error than they remove on these races (practice
long runs that are not race-like: HARD got faster in practice at 2019 United Kingdom and 2022
Saudi Arabia; qualifying makes SOFT about 1 s faster than MEDIUM where the race fits put them
within 0.3 s). Both are far better than the chain, which put practice first.

**Decision level**: for each race, each raced compound hidden in turn and one lap-0 snapshot
per starting compound (a finisher's), the reference (`optimize_batch`, seed 0, 50 samples,
two stops) on the world with that compound hidden and proxied, against the reference on the
true world (`validation_decisions.csv`, `validation_decisions_summary.csv`). Cost is the true
world's time of the proxied world's plan less the true optimum (valid rows):

| priced by | rows | plan changed | invalid in the true world | cost, s: mean / median / max | laps on the hidden compound: more / fewer / same |
| --- | --- | --- | --- | --- | --- |
| (i) the chain as built (old step table) | 47 | 42 | 4 | 2.77 / 0.69 / 23.56 | 12 / 23 / 12 |
| (ii) cross-race step alone | 47 | 40 | 3 | 3.45 / 2.01 / 13.52 | 12 / 24 / 11 |
| (iii) the blend, leave-one-race-out (shipped) | 47 | 41 | 3 | 3.57 / 2.01 / 13.52 | 11 / 26 / 10 |

**The step alone beats the blend at decision level too, by 0.12 s mean cost (within 0.3 s),
and the chain as built had the lowest mean (2.77 s) and median (0.69 s) cost but the worst
tail (23.6 s, practice HARD at Saudi) and one more invalid row.** The three are within about
1 s of each other on 47 cases; the parameter errors above separate them more clearly. In all
three the search uses the hidden compound less than the truth more often than more, so it does
not systematically exploit a proxied compound. The invalid rows run the hidden compound past
its true race limit, because its proxied limit is its practice age (2023 Bahrain MEDIUM,
raced to 10 laps; 2024 Emilia Romagna SOFT, raced to 8; for the step alone also 2022 Saudi
Arabia MEDIUM, raced to 16).

### What changed in the references

References rebuilt with the blend at seed 0, 50 samples, two stops (`python -m src.eval
reference`), against main's. With the proxy off every record of every set is identical to
main's (rebuilt in a scratch directory and compared column by column); with it on, only races
with a proxied compound move:

| set | references | plan changed | plan on a proxied compound | stops 0 / 1 / 2, main | stops 0 / 1 / 2, blend |
| --- | --- | --- | --- | --- | --- |
| v1 (2021 Spain) | 264 | 66 | 66 (HARD) | 40 / 216 / 8 | 40 / 174 / 50 |
| pilot1 (2021 Spain) | 25 | 7 | 7 | 3 / 22 / 0 | 3 / 15 / 7 |
| d2 (2023 Spain) | 25 | 0 | 0 | 8 / 17 / 0 | 8 / 17 / 0 |
| dev6 (six races) | 150 | 7 | 7 (Spain 7 of 25, Saudi 0 of 25) | 56 / 85 / 9 | 56 / 78 / 16 |

The two real cases, on main, with the chain as built (#81's first version) and with the blend
(dev6 baselines re-scored in a scratch directory; regret is at least 0 on every scored row):

| | unraced compound | blended fresh-pace step from the anchor | blended wear | dev6 references using it | actual_strategy mean regret, s: main / chain / blend |
| --- | --- | --- | --- | --- | --- |
| 2022 Saudi Arabia | SOFT | -0.21 s from MEDIUM (practice 8%, step 92%) | 0.127 s/lap per lap (MEDIUM 0.064; practice 29%, step 71%) | 0 of 25 (chain: 25 of 25) | 0.72 / 7.71 / 0.72 |
| 2021 Spain | HARD | +0.28 s from MEDIUM (step only; raised to SOFT's base) | 0.033 (lowered to SOFT's) | 7 of 25 | 8.91 / 9.00 / 9.08 |

At Saudi the blend leaves the references where main had them: SOFT, 0.2 s faster than MEDIUM
fresh but wearing twice as much, is not worth a stop. At 2021 Spain HARD is priced by the step
alone (no HARD long run, no HARD in qualifying), and the ordering constraint makes it a copy of
SOFT: the race fit has SOFT both slower than MEDIUM fresh and wearing less, so HARD may be
neither faster than SOFT nor wear more than it. The references then use HARD as a second
SOFT-like compound (SOFT-HARD-SOFT two-stops, capped at 21 laps by R8), which a SOFT-SOFT-SOFT
plan could not do under R3: 66 of v1's 264 references and 7 of dev6's 25 Spain ones. Over dev6
the mean regret moves from 6.97 to 7.15 s (actual_strategy), 8.21 to 9.00 (fixed_one_stop, which
takes the proxied HARD at Spain and is invalid on 13 more rows past its practice cap),
24.38 to 24.44 (fixed_two_stop) and 22.01 to 20.86 s (greedy_crossover). Replaying the dev6
pass-1 agent submissions (no agent run) on the blend: no row is invalid for an unknown compound;
Saudi Arabia goes from 21 invalid rows to 1 (a PIT the minimum-stint rule refuses).

**Limitation.** A proxied compound's price has no race laps behind it: its pace and wear are
the anchor's plus a step measured elsewhere (practice, qualifying, other races), with the
errors above, and its stint limit is a practice age. Read `proxied_compounds` and
`opt_proxied_compounds` beside any result on a race with a proxied compound.

## Historical tyre-life prior

`src/sim/tyre_history.py` estimates where a compound's tyre life runs out at a race (the
simulator's support edge) from **earlier races only**, instead of from the race being evaluated.
Nothing in the simulator calls it yet (it is `NON_PHYSICS`); the `fix/tyre-life-no-hindsight`
change will place the edge with it in both the belief (this prior, extended by tyre ages seen up
to the snapshot) and the world (this prior, extended by tyre ages the race's laps show), and must
then move the module to `PHYSICS_MODULES` or carry its numbers into `SimConfig`.

```python
from src.sim.tyre_history import tyre_life_prior
p = tyre_life_prior("2023_07_spain", "SOFT")     # role: HARD / MEDIUM / SOFT
p.edge_laps, p.sd_laps, p.n_effective, p.tier, p.sources, p.compound, p.interval(0.9)
```

**Inputs, all committed, none loaded from a race at run time.**
`data/derived/tyre_nominations.csv` (Pirelli's nomination per weekend, public pre-race
information, two sources per row; `data/derived/README.md`), `data/derived/compound_scale.csv`
(every season's C-numbers on one hardness index), `data/derived/tyre_history.csv` (the per-race
observations, built by `python -m src.sim.tyre_history build` from every cached race not held
out; `tyre_history_manifest.json` lists them with their dates) and
`configs/tyre_history.yaml` (every constant below).

**As of, and held out.** Only races strictly before the target count: earlier by (season, round)
(`race_ids.as_of`) and, where both dates are known, by date. A race in `data/eval_set.json` (or
the config's `held_out_extra`) is never built into the history and never feeds the prior; the
`allow_held_out` flag (default off) lets in only a held-out race that is itself earlier.

**Compound scale.** The meaning of a C-number moved: 2019-2021 C1..C5 (one set of compounds),
2022 C1..C5 (new 18-inch compounds, same names), 2023 C0..C5 (the old C1 renamed C0, a new C1
"much closer to the C2"), 2024 C1..C5 (C0 dropped, compounds unchanged), 2025 C1..C6 (a C6 softer
than the C5). The common index counts steps of the 2019-2022 scale, larger is softer: Cn -> n,
except the 2023-2025 C1 -> 1.5, the 2023 C0 -> 1 and the 2025 C6 -> 6. Sources in
`data/derived/README.md`.

**The observation** (`observed_tyre_life`), per race and compound: the stints are split by how
they ended. A pit stop on a green in-lap and a stint run to the chequered flag are *usable*: the
tyre lasted at least that long, and the team chose when to stop. A stop under SC / VSC / red flag
(opportunistic), a red-flag tyre change and a retirement say nothing about tyre life and are left
out. The edge is the **0.9 quantile of the usable stints' end ages** (at least 3 stints), in laps
and as a share of the race distance: a high quantile, because teams stop before the tyre is
finished, but not the maximum, so one car's gamble does not set it. A pace check is reported next
to it (per stint with 10+ green laps, the excess of its last 3 laps over a straight line through
its first 60%: `pace_late_excess_s`, `pace_falloff_share`) and never used. 2025 Qatar's
observation is kept but excluded (a mandated 25-lap maximum stint); a race with more than 10% of
its laps on intermediates or wets is down-weighted.

**The prior.** A same-circuit estimate shrunk toward a pooled one, never a switch between them:
- *pooled*: every usable earlier observation, each treated as another circuit's, moved onto the
  target as a share of race distance (times the target's scheduled laps); for another compound,
  times `life_ratio ** steps` (`steps` = target index - observed index, one step softer lasting
  `life_ratio` as long; the ratio is the geometric mean over the earlier races' compound pairs,
  else 0.75); plus the circuit's severity as Pirelli judged it before both races, the mean index
  of the nominated trio (a circuit given a trio one step softer is one step gentler). For the same
  role at two circuits this cancels the C-number difference. Weighted (constants, not fitted):
  same compound 0.1, other compound 0.03, times 0.5 per further index step, 0.85 per season
  back, 0.4 from the other regulation era (2018-2021 vs 2022+), 0.3 from a wet race.
- *same circuit*: that circuit's own earlier observations, not moved by severity, weighted same
  compound 1.0, other compound 0.3, with the same step, recency and era factors. A race flagged
  **abnormal** (a red flag, or more than 2% of its laps on intermediates or wets: flags the
  history carries, known once that race is over) is left out of it (`abnormal_same_circuit` 0)
  and stays in the pool.
- *combined*: `w_s * same circuit + (1 - w_s) * pooled`, `w_s = n / (n + k0)`. `n` is the
  circuit's effective number of dry, normal earlier races (each race counts its heaviest
  observation's weight relative to a same-compound race of the same season: a same-compound race
  one season back counts 0.85), and `k0 = 2` races is the stated prior strength of the pool
  (`shrinkage.prior_strength_races`, not fitted). One earlier race at a circuit therefore moves
  the estimate about a third of the way from the pool.
- *sd*: the two-part mixture's: `w_p var_pool + w_s var_local + w_p w_s (local - pooled)^2`
  (each part's weighted spread with the Kish `n_effective`), so it widens when the same circuit
  and the pool disagree; in quadrature with a floor blended the same way between the tier's and
  tier 3's (0.05 / 0.08 / 0.10 / 0.20 of race distance for tiers 1 / 2 / 3 / 4).

**Tier** (always returned): the most specific source that contributed. 1 a normal earlier race
at the same circuit with the same compound; 2 the same circuit, other compounds only; 3 the pool
only; 4 no usable earlier race at all, a constant share of race distance per role (HARD 0.65,
MEDIUM 0.50, SOFT 0.35). The earliest cached season is tier 4. A missing nomination (the
target's or an earlier race's) compares compounds by role and never counts as the same compound.
`sources` lists each observation's share of the estimate, largest first.

**Validation** (`python -m src.sim.tyre_history validate`, `reports/calibration/tyre_history/`).
As of, hence leave-one-race-out, on the 63 usable (race, role) observations of the 27 cached races
with a dry observation that are not held out (29 races built: 2021 Belgium and 2024 Brazil have
no dry observation; 2025 Qatar is excluded; the held-out races of `data/eval_set.json` are never
loaded). Error = prior edge - observed edge, in laps; coverage of the stated 68% / 90%
intervals; *beyond* = share of the race's usable stints on that role ending later than prior
edge + 3 laps (*floor*: the same against the race's own observed edge, what a perfect prior would
give). *Normal races* leaves out the 15 observations whose own race was red-flagged, wet or damp
(2021 Imola and Saudi Arabia, 2022 Imola, Monaco and Singapore, 2024 Monaco and Silverstone):
no prior can know that before the race.

| group | n | mean error | MAE | cover 68 | cover 90 | stints | beyond | floor |
|---|---|---|---|---|---|---|---|---|
| all | 63 | -0.6 | 7.0 | 0.71 | 0.89 | 1004 | 0.156 | 0.029 |
| tier 1 | 14 | +3.9 | 4.1 | 0.86 | 1.00 | 327 | 0.009 | 0.031 |
| tier 2 | 4 | +10.1 | 10.1 | 0.50 | 0.75 | 57 | 0.018 | 0.018 |
| tier 3 | 42 | -3.0 | 7.9 | 0.67 | 0.86 | 583 | 0.256 | 0.027 |
| tier 4 | 3 | -2.6 | 2.9 | 1.00 | 1.00 | 37 | 0.108 | 0.054 |
| era 2018-2021 | 21 | -1.6 | 6.0 | 0.76 | 0.86 | 301 | 0.199 | 0.027 |
| era 2022+ | 42 | -0.1 | 7.5 | 0.69 | 0.90 | 703 | 0.138 | 0.030 |
| normal races | 48 | +1.5 | 5.7 | 0.79 | 0.94 | 851 | 0.107 | 0.033 |

Against the earlier tier-switching prior (same 63 observations): MAE 7.0 both; 68% / 90%
coverage 0.70 / 0.84 -> 0.71 / 0.89; stints past edge + 3, 14.9% -> 15.6%; on normal races MAE
5.7 both, 90% coverage 0.92 -> 0.94. On the 22 cases the switching prior called tier 1, MAE 8.4
-> 8.3 and 90% coverage 0.77 -> 0.91. The tiers are not the same sets (8 of those 22 now have no
normal same-circuit race and are tier 3), so per-tier rows do not compare across the two. The
shrinkage hardly moves the mean because the switching prior's same-circuit weight (1.0 against
0.1 per other race) already gave one same-circuit race about a third to a half of the estimate;
what it fixes is the stated uncertainty (the disagreement term) and the abnormal-race rule. The
largest errors were never the earlier race's fault: they are races that were abnormal themselves
(2024 Monaco's lap-1 red flag, medium run 76 laps against a prior of 36; 2022 Imola, damp, medium
46 against 30).

The weighting, each part switched off in turn (all 63 / the 48 normal):

| variant | MAE | mean error | cover 90 | beyond | MAE, normal races |
|---|---|---|---|---|---|
| full (as shipped) | 7.0 | -0.6 | 0.89 | 0.156 | 5.7 |
| no circuit severity scaling | 9.2 | -2.3 | 0.87 | 0.216 | 7.0 |
| no C-number (compounds compared by role) | 7.1 | -1.0 | 0.87 | 0.165 | 5.7 |
| no same-circuit part (pool only) | 7.0 | -0.6 | 0.89 | 0.157 | 5.8 |
| neither C-number nor circuit | 7.2 | -0.9 | 0.87 | 0.166 | 5.9 |

Reading it: the severity scaling is the one part that clearly pays (9.2 -> 7.0 laps); matching
the bare C-number across circuits without it is the worst variant, because Pirelli nominates
harder rubber for harsher circuits, so a C3 at Imola and a C3 at Silverstone are not run alike.
The C-number and the same-circuit part are each worth about 0.1 lap here. 16% of real stints run
past edge + 3 laps against a 3% floor (11% on normal races): the prior is short on the
long-running side, mostly at tier 3 (26%) and in 2021 (20%).

Before the 2021-22 weekends of `data-more-races` (#92) were added the validation had 20
observations from 9 races: MAE 8.2, 90% coverage 0.90, 11.4% of stints past edge + 3. The
weights and the prior strength stay as stated (they are not tuned to these tables).

## The agent policy

`--policies agent` is `eval.policies.AgentPolicy`, the snapshot loop of `src/agent.py`, on
any scenario of any race (the lap-0 same-race guard is gone):

- the agent's view is `RaceSnapshot.from_contract(scenario.snapshot)` (`src/snapshot.py`):
  lap, tyre, gaps, `field.nearby`, `recent_lap_times_s`, weather, `race.min_stint_laps`
  and the inventory-legal compounds; the stint history and the sets stay out of it;
- the tools keep the contract itself (`tools.set_snapshot`) and answer from the belief
  simulator at the snapshot (`src/sim/agent_tools.py`, SPEC 0015): degradation and pit
  loss ranges, the model report, plan comparisons, and an action against its alternative,
  with the gap between them read from the belief's paired draws ("Plan gaps and rule
  replies" below); the times include traffic from the projected field, and the outputs
  carry it compactly ("Traffic" below);
- `run_agent(scenario.snapshot, verbose=False, max_steps=profile.agent.max_steps, log_path=...)` logs
  the trajectory to `reports/eval/trajectories/<set version>/<scenario_id>.jsonl` (":" in
  the id written "_"; a rerun replaces the file);
- the submission is scored as a decision (above); `tool_calls` is the trajectory length
  (every attempted call: refused ones, repeats and submission attempts included),
  `evidence_calls` the evidence calls that actually ran (no submissions, no calls refused
  for the budget or in a grace turn, no repeats served from the cache; a repeat that was
  run again counts, since it ran), `repeat_calls` the evidence calls identical to an
  earlier one in the episode (same tool and arguments), whether run again or served from
  the cache, so looping stays measurable in either repeat mode (a repeat refused for the
  budget or in a grace turn is not counted), `input_tokens` and `output_tokens` the
  backend's reported usage (blank when it reports none); a backend exception is an `error`
  row, not a crash;
- `AGENT_REPEAT_MODE` (also `run_agent(repeat_mode=...)`) says what a repeat gets: `run`
  (the default) runs it again and counts it against the evidence budget, as before #72;
  `cache` gives it the earlier call's cached result with `repeat_of` added and uses no
  budget, also once the budget is spent (a grace turn still refuses it). Any other value
  is an error. In both modes the trajectory event carries `repeat_of` (the earlier
  call's `call_index`); in `run` mode the model's tool result does not;
- `AGENT_TURN_STATUS` (`1`, `true` or `on`; also `run_agent(turn_status=...)`) ends each
  evidence turn's tool-results message with "Turn s of N. Evidence calls used u of B."
  and, on turn N-1, "One evidence turn is left after this one." It is off by default:
  the tool-results messages are then what they were before #72;
- each trajectory event has a `phase`: `evidence` for the `max_steps` turns, `grace` for
  the up to `AGENT_SUBMIT_GRACE_STEPS` (default 2) turns that follow when no submission was
  accepted, which offer only `submit_recommendation`; `submitted_in_grace` marks a
  submission made there. The grace turns are on whatever the two options above say;
- a turn whose reply holds no usable tool call is a line in the trajectory file with
  `event: "no_tool_call"` (step, phase, the reply clipped to 1500 characters, the backend's
  parse error, finish reason and output tokens), not a trajectory event, so `tool_calls`
  does not count it. The next message tells the model what was wrong: a reply cut off at
  the output cap, or the parse error with the text around where it failed (a reply that
  simply has no call still gets "No tool call was detected..."). The Voyager backend reads
  a reply whose JSON fails only in its final run of closing brackets as the one run that
  closes what is open (everything before it well nested and kept as written; never a reply
  cut off at the cap); the calls from such a reply carry a `parse_repair` note. dev6
  run 4 is the first pass with this; up to dev6 run 3 and in the held-out d4 pass these
  turns were not logged, the reminder was the bare one and nothing was repaired. Since
  2026-11-22 it also reads two more shapes, each with its `parse_repair` note, and guesses at
  nothing malformed: a reply of two or more complete `{"tool_calls": [...]}` objects, each
  valid on its own, with only whitespace or bracket-free prose around them, as their calls
  in order (the per-turn and budget rules apply to them as to one envelope); and a bare
  JSON object whose keys are exactly a valid `submit_recommendation` argument set (every
  required key, nothing undeclared) as that call.

### Plan gaps and rule replies

- **Gaps between plans.** `AGENT_GAP_YARDSTICK` says how the tools judge a gap between two
  plans. `paired` (the default since 2026-11-22): `evaluate_strategy` gives `gap_reading`, e.g.
  "STAY_OUT 1 lap is faster than PIT MEDIUM in 82% of draws; STAY_OUT 1 lap minus PIT
  MEDIUM: -3.1 s (90%: -6.0 to -0.2 s)". The share is `probability_improves_outcome`; the
  gap and its interval are the median and 5th / 95th percentiles of the action's time minus
  the alternative's over the same draws (common random numbers, R6). `compare_plans` adds
  `gap_readings`: the lowest-mean candidate against each other valid one, with `p_beats`
  and the interval mean -+ 1.64 x `gap_std_s`. These intervals are the model's own: they
  cover parameter and lap-noise uncertainty under the belief and not model error. No
  cross-race calibration of gap error exists as of a race (the fidelity scorecard's B3
  `gap_cov_90`, the share of judge gaps to the real plan inside the belief's 90% paired
  interval, was 0.69 pooled over 18 development races on code 3ed6e2af6d0b, so the
  interval is likely too narrow; it is pooled over later races, so the tools do not read
  it). `calibration_error_s`, the error of the absolute remaining time measured on one
  race (16 to 36 s), is not used for gaps: errors common to both plans (fuel, pace offset,
  the track) cancel in a difference. `get_model_report` still shows it, with a note
  saying so; `evaluate_strategy` and `estimate_tyre_degradation` no longer carry it.
  `calibration` restores the legacy outputs (`within_sim_error` against
  max(2 sd, `calibration_error_s`), `confidence_basis`, the calibration fields and the old
  tool and prompt text) to reproduce earlier runs. The tool descriptions follow the
  switch as the tools are imported; any other value is an error. Rerun provenance records it.
- **Rule replies.** A plan the belief's rules reject comes back, in `compare_plans` and
  `evaluate_strategy`, with each minimum-stint, two-compound and lap-count message
  restated (`agent_tools.explain_rule_errors`; `rules.py`, a physics module, is unchanged):
  which stint and why, and the nearest legal alternative, from the snapshot and the
  regulations only (laps so far, laps on the set, the scheduled distance, the minimum).
  For example, a pit at lap 0 of 66 with a 3-lap minimum: "Below the 3-lap minimum stint: stopping at the end of lap 1 makes the current SOFT stint 1 lap
  (0 run so far + 1). The earliest legal stop is at the end of lap 3 (stay out 2 laps
  first, then pit), the latest at the end of lap 63." Three laps from the end: "the final
  stint (MEDIUM) would be 2 laps. No stop fits in the 3 laps left: only STAY_OUT to the
  flag is legal." A single-compound plan gets the legal stop window and the longest
  `stay_out` that keeps a stop possible; a lap-count mismatch gets the laps that remain and
  the last-stint length that would add up. No tool reply exists for too many stops: the
  stop limit only bounds the optimizer's continuation, which then runs to the flag and
  gets the single-compound reply. Other messages pass unchanged.
- **Confidence.** The system prompt defines the submitted `confidence` in one sentence as
  the probability that the recommended action is the better one (PIT over STAY_OUT, or the
  reverse), which is how `p_pit` scores it.

Why those defaults: in live runs on the same 50 scenarios, unprompted submissions went 43
with neither the status line nor repeat handling, 31 with the status line and refused
repeats, and 27 with the status line and cached repeats, while input
tokens per episode went 53k, 63k, 85k. The decisions barely changed. The grace turns are
what guarantee a submission.

`src.grade.grade_decision(snapshot, submission)` grades one submission the same way (valid,
regret against the optimizer from that snapshot) for `tests/test_case.py`, which reports
the regret and does not assert it.

## Scenario sets

A scenario is one (race, driver, lap) snapshot plus the driver's actual full-race stints. A
set is sampled once and written with every snapshot in full, so changes to the snapshot
builder, the cache or the lap filter never move a published set. The file carries a sha256
of its items and `ScenarioSet.load` refuses a file whose items no longer match: build a new
version instead of editing one.

**v1** (`data/scenarios/v1.json`, 264 scenarios): 2021 Spain; the 19 drivers with laps 1..n
and at most two laps down (Tsunoda's DNF is out); the decision laps of
`calibration.default_laps_k` (every 5th lap from 0, plus the laps either side of each SC and
VSC period) that leave at least one minimum stint to run. Snapshots come from Jason's
stand-in builder `src.sim.calibration.snapshot_at`.

v1 snapshots carry `race.min_stint_laps` 10 (the rule before R2 became 3), so v1 keeps
scoring under 10: the committed split table reproduces row for row. Sets sampled from now
on carry the registry's value, 3 by default (docs/rules.md R2).

Next versions: v2 once Sarah's snapshot builder and the new races land (pass it as
`builder=` to `build_set`; it has the same signature as `snapshot_at`).

Every set listed here was sampled under the legacy rule, which reads the future (above,
"Information contract"). New sets default to `--rule as_of` and the builder
`calibration.snapshot_as_of`; `--rule legacy` reproduces the old sampling. One sampler serves both
new rules: `as_of` takes `scenarios.as_of_pool` every 5th lap, `stratified` (below) the same pool
every 2nd lap, and both build each snapshot with `snapshot_as_of`.

### Decision-critical sets

`python -m src.eval scenarios --rule stratified ...` (`scenarios.sample_stratified`) samples a
set where the decisions are. On dev6, 120 of 150 scenarios are ties between the main policies and
the differences sit in the other 30, and pit-now is optimal at 7% of the uniform set's snapshots
(the design study raised it to 24% with this rule). No committed set changes; none is built here,
and the rule is never to be run on held-out races.

- **Pool**: every 2nd lap L from 0 to the scheduled race length less the minimum stint, for the
  drivers running at L (laps 1..L all present). It reads no lap after L and no final lap count
  (the legacy rule's finisher filter and lap cap read the race's end). It is `scenarios.as_of_pool`, the
  as_of rule's pool at a finer step.
- **Strata**, the first that applies, in order:
  `pit_opt` (the world's reference pits at the end of the next lap), `close` (not pit_opt; the best
  pit-now and best stay-out plans within 3 s, and the worst searched plan at least 10 s behind),
  `neutral` (the driver's lap L under SC or VSC, or one of the 2 laps before it), `window` (some car
  pitted on laps L-2..L), `rest`. `neutral`, `window` and the pool read only the race's laps up to
  L. **`pit_opt` and `close` read the scorer's reference**, which is future information: that is
  acceptable only because scenario selection is invisible to every policy. The stratum and the
  weight are recorded on each item (`Scenario.stratum`, `Scenario.weight`) and in the scored rows,
  and `score.policy_scenario` drops both before any policy, tool or snapshot sees the scenario
  (tests/test_sim_neutralised_world.py). Nothing depends on a policy's result.
- **Quotas** per race (`DEFAULT_QUOTAS`): pit_opt 7, close 6, neutral 4, window 4, rest 4; a
  shortfall goes to rest; seeded per race and stratum. **Weights**: stratum pool size / sampled,
  so a pooled mean reweights to the uniform pool (the weights sum to the pool size); report per
  stratum and the weighted pooled mean. The set records the rule in `sampling`; the references it
  stratified on are the world's at build time (its config hash is the scored rows' `world_config_hash`).

### Development and report sets

`data/eval_set.json` is the registered eval split, written by `register_eval_set` (Sarah's
item): the races held out for reporting, `{"races": [...], ...}` or a bare list of ids.
While it exists, `build_set` (and `scenarios`, `--eval-set` for another file) refuses any
race outside it, records the file's sha256 as `sampling["eval_set_digest"]`, and marks the
set `development: false`. Without it a set is `development: true`.

- **v1 is the development set**: sampled before the split, from the race we tuned on.
  Its numbers are for debugging the harness and the baselines, not for the report.
- **A set sampled from the eval races is the report set**: the numbers the report quotes.
  Today that is d4, built and scored once at tag `d4-final-eval` (see "The registered split"
  below and "d4, the held-out result").

`scenarios --development` (`build_set(..., development=True)`) still samples a development
set once the split exists, a Spain debug set say: no race is refused, the set is marked
`development: true`, and `sampling["eval_set_digest"]` is recorded anyway so the set shows
the split existed. Report numbers come only from sets with `development: false`; `score`
prints one line naming which kind of set it scored.

### The registered split

The cache holds 2021 Spain plus a 2018-25 sample (dataset 3): the 2023 and 2024 races already
there and eight drawn with `random.seed(2026)` from the 173 races of 2018-25, 14 race weekends
in all. (24 weekends of 2021-22 were added later, 38 in all; see "More cached weekends" below.
Five of them joined this split on 2026-11-21; see "The fresh 2021-22 draw" below.) Of the 14, 2024 Monaco (lap-1 red flag), 2024 Britain (wet) and 2024 Sao Paulo (wet and
red flag) are not eligible.

`data/eval_set.json` holds out nine races. The first four are the d4 split: **2019 Monaco,
2020 Sakhir, 2025 Austria and 2024 Austria**. Five 2021-22 races were registered on
2026-11-21: **2021 France, 2021 Mexico, 2022 Miami, 2022 Australia and 2022 Belgium** ("The
fresh 2021-22 draw" below). Nothing loads them until the final run: no exploratory commands,
no fits, and no test that asserts on their content. Their eligibility records are the one
exception (2024 Austria below, the five in "More cached weekends").

| race id | role |
|---|---|
| 2019_06_monaco, 2020_16_sakhir, 2025_11_austria, 2024_11_austria | held out (`data/eval_set.json`, the d4 split) |
| 2021_07_france, 2021_18_mexico, 2022_05_miami, 2022_03_australia, 2022_14_belgium | held out (`data/eval_set.json`, the fresh 2021-22 draw) |
| 2021_04_spain | development and calibration: the simulator was tuned on it |
| 2023_07_spain | calibration (held out under the previous split, see below) |
| 2023_01_bahrain, 2019_10_united_kingdom, 2022_02_saudi_arabia, 2024_07_emilia_romagna | calibration |
| 2025_23_qatar | calibration only, never sampled into an eval scenario set |
| 2024_08_monaco, 2024_12_united_kingdom, 2024_21_brazil | not eligible |
| the 13 other eligible 2021-22 races ("More cached weekends") | calibration and history |
| 2021_02_emilia_romagna, 2021_21_saudi_arabia, 2022_04_emilia_romagna, 2021_12_belgium, 2022_07_monaco, 2022_17_singapore | not eligible (practice and history only) |

How the four were chosen:

- **Sarah's draw, extended.** The pool is the six newly cached eligible races, in season and
  round order. Sarah's registered draw took one race from it; the same draw with k raised
  from 1 to 3 gives three:

  ```python
  pool = ["2019_06_monaco", "2019_10_united_kingdom", "2020_16_sakhir",
          "2022_02_saudi_arabia", "2024_07_emilia_romagna", "2025_11_austria"]
  random.seed(2026); random.sample(pool, 3)   # ['2019_06_monaco', '2020_16_sakhir', '2025_11_austria']
  ```

  With k=1 the same seed returns `['2019_06_monaco']`, Sarah's draw. 2019 Monaco is kept
  although track position dominates there.
- **2024 Austria is added** because it is the one eligible race no run has ever loaded. It is a
  sprint weekend, so real-run discovery left it out until sprint weekends were supported.
  Eligibility facts only, from the race session, with no model run: dry, no red flag, 19
  eligible drivers, HARD, MEDIUM and SOFT raced, 71 laps.
- **2023 Spain moves to calibration.** It was the held-out report race of d2 and d3, and it was
  used to diagnose the simulator after Milestone 2, so it can no longer be held out.

Disclosure: what each held-out race has been exposed to.

- **2019 Monaco:** references and the four baselines were scored once on Nov 11 under the
  pre-R8 simulator (Sarah's dataset 3 commit 4a8477c; `reports/eval/d3_baselines.csv` has 100
  Monaco rows at sim_code_version b62094155bde and no agent rows) and reported. No agent run.
  No modelling choice used those numbers.
- **2020 Sakhir and 2025 Austria:** two of the seven races in the leave-one-race-out comparison
  of Nov 12 that pointed to driver effects with the hinge and to rule R8, so they informed
  those choices before being held out. No eval scenarios, no baselines, no agent run.
- **2024 Austria:** nothing at all. Eligibility facts only.
- **All four:** the agent has never been run on any of them.

The fresh 2021-22 draw (registered 2026-11-21). The final test set is the four races above plus
a season-stratified draw from the 2021-22 weekends the more-races cache added. Results can then
be shown across regulation eras: 2021 is the previous car generation, 2022 the first
ground-effect season, and 2024-25 later ground-effect seasons. The pools are the eligible
group-B races (the seeded pool, not the circuit-history group A), in season and round order.
The draw uses one seed, with 3 races per season, or 2 when a season has fewer than 5 eligible
races. 2021 has 3, so it gives 2:

```python
pool_2021 = ["2021_07_france", "2021_18_mexico", "2021_22_united_arab_emirates"]
pool_2022 = ["2022_03_australia", "2022_05_miami", "2022_08_azerbaijan", "2022_09_canada", "2022_12_france",
             "2022_14_belgium", "2022_20_mexico", "2022_21_brazil", "2022_22_united_arab_emirates"]
random.seed(2027)
random.sample(pool_2021, 2)   # ['2021_07_france', '2021_18_mexico']
random.sample(pool_2022, 3)   # ['2022_05_miami', '2022_03_australia', '2022_14_belgium']
```

All five are conventional weekends. Exposure: nothing beyond the eligibility facts in "More
cached weekends". They were cached and checked offline: the race id resolves, the sessions
load, and `calibration.eligibility` ran on the race session. There was no fit, calibration,
cross-race pass, scenario, reference, baseline or agent run, and neither the tyre-history
table nor the pit-loss prior holds them. From 2026-11-21 every held-out guard, `--races
auto` and the tyre-history build leave them out, and `build_set` accepts them for a report set.
The file's sha256 is
now `4f96f7f3c37a...`, where it was `98edff4380e1...`.

Scenario sets under the split:

- **d2** (`data/scenarios/d2.json`, 25 scenarios of 2023 Spain) and **d3**
  (`data/scenarios/d3.json`, 50 scenarios: 25 each of 2019 Monaco and 2023 Spain, seed 0) were
  sampled under the earlier splits, and each keeps the `eval_set_digest` of the file it was
  sampled under (d2 the one-race split `{"races": ["2023_07_spain"]}`, d3 the two-race split
  of 2019 Monaco and 2023 Spain). Both are records: they still load and are not regenerated.
  d3 is the Milestone-era set and is no longer the report set. Since half of it is a held-out
  race, no new references, baselines or agent runs are made on d3. Its Monaco baseline rows
  stay in the repo as they are, are not regenerated, and are superseded by d4 at the final run.
- **d4** is the report set: the four held-out races, 25 scenarios each, seed 0,
  `development: false`. Its scenario file is built only after sprint-weekend support lands
  (2024 Austria is a sprint weekend), and no references, baselines or agent runs are made on
  it until the final run. d4's Monaco scenarios may coincide with d3's, since both sample 25
  per race at seed 0; the final run will state whether they do. `data/scenarios/d4.json`
  records the four-race split's `eval_set_digest` (`98edff4380e1...`), not the nine-race file's.
  Like d2 and d3, it is a record of the split it was sampled under, and it still loads. It is
  not regenerated.
- **dev6** (`data/scenarios/dev6.json`, 150 scenarios, digest `8823b7b73c87`) is a
  **development set on calibration races, not a held-out result**: 25 scenarios each of the six
  non-sprint calibration races 2021_04_spain, 2023_07_spain, 2023_01_bahrain,
  2019_10_united_kingdom, 2022_02_saudi_arabia and 2024_07_emilia_romagna, seed 0,
  `--development` (`scenarios --races <the six> --version dev6 --max-per-race 25 --seed 0
  --development`). The simulator was calibrated on these races, so its numbers say how the agent
  and the baselines behave where the model fits best; they are not a report-set estimate. Its 25
  2021 Spain scenarios are pilot1's and its 25 2023 Spain scenarios are d2's (same ids and the
  same snapshots, item for item). References: `data/references/dev6/fbb33ef732dd.parquet`
  (sim_code_version `58b50a00e0e3`, seed 0, 50 samples, two stops, default world with the
  compound proxy's blend). Pass 1 and `dev6_baselines.csv` were scored on the earlier
  `4ecc62a30937` (sim_code_version `1dcf46ad75aa`), before the compound proxy, where 2022 Saudi
  Arabia had a known issue: the belief modelled SOFT (from the practice prior) but the world,
  fitted on the race laps, modelled only HARD and MEDIUM, so a decision whose belief
  continuation used SOFT was scored invalid with "Unknown compound 'SOFT'" (20 of the agent's
  21 invalid rows on pass 1, and 5 greedy_crossover rows). The proxy prices SOFT in the world
  now ("Proxied compounds"); the committed pass-1 and baseline files are not rescored here.

### Sprint weekends

A sprint weekend (2023 on) has one practice session, FP1, and a Sprint on Saturday. The practice prior
reads the FP sessions a weekend has plus the Sprint (`calibrate.practice_sessions`, from the
cached event schedule): FP1 and the Sprint on a sprint weekend, for both the degradation prior
(`calibrate._practice`) and the pace offsets (`pace.load_practice`, as `cross_race` and the
`calibrate` commands call it). A conventional weekend reads what it always did, FP1-FP3 for
the prior and FP2-FP3 for the pace offsets, and its practice frame is unchanged (checked frame
for frame on 2021 Spain and 2023 Bahrain). No physics module changed, so `SIM_CODE_VERSION`
and the cached references are untouched. Real-run
discovery (`real_run.cached_races`) now lists the three cached sprint weekends, 2024 Austria,
2024 Sao Paulo and 2025 Qatar, where it used to leave them out with a note. Listing is not
loading: 2024 Austria is held out, and 2024 Sao Paulo is not eligible.

- On a sprint weekend both the prior and the pace offsets read FP1 + Sprint: FP1 is the only
  practice session, and the Sprint alone would tie a driver's pace to where they ran in the
  train. This choice was fixed before any sprint race was evaluated.
- 2021-2022 sprint weekends read FP1, FP2 and the Sprint by the same rule. The session
  selection is unit-tested. Three 2022 sprint weekends are cached now (2022 Emilia Romagna,
  Austria and Brazil, "More cached weekends" below), but the practice prior and the pace
  offsets have not been run on any of them yet.

**2025 Qatar is used for calibration only and is never sampled into an eval set.** The
weekend had a mandated maximum of 25 laps per tyre set. Race control messages do not state
it (it was set before the event), but the stint data does: no car ran a lap on a tyre older
than 25 laps on any compound in FP1, the Sprint or the race; 29 of the 61 race stints end at
exactly 25 laps of tyre life; 16 cars pitted under the lap-7 safety car and 15 of them pitted
again on lap 32, 25 laps later, to finish the 57 laps on a second 25-lap set. Rules R1-R8 have
no rule for a stint cap, so a recommendation scored there could break a rule the race had.
`build_set` refuses it (`scenarios.CALIBRATION_ONLY`) in every mode, `--development`
included.

### More cached weekends (2021-22)

24 race weekends of 2021-22 were added to `cache/`. They give the simulator calibration races
beyond the 7 it had, and more than the 8 that `MIN_HISTORY_RACES` needs. They add earlier
history at circuits already in use, for as-of tyre-life and pit-loss priors. They also give a
pool of races that no run has loaded, from which a fresh held-out draw can be made. Each
weekend was built with `python scripts/build_cache.py --season <y> --gp "<event name>"
--sessions <sessions> --prune` (Sarah's builder: no telemetry, raw livetiming responses
pruned from the sqlite, then VACUUM). The sessions are every session in the event schedule:
FP1, FP2, FP3, Qualifying and Race on a conventional weekend, and FP1, Qualifying, FP2, Sprint
and Race on a 2022 sprint weekend. Afterwards each weekend was checked offline: the race id
resolves from the cached schedule (`data.schedule_race_id`), and every session loads under
`fastf1.Cache.offline_mode(True)` (`data.load_session`). Size: 41.0 MB of session pickles
(0.6-2.2 MB per weekend), and the sqlite grew from 524 KB to 1.25 MB with the Ergast results
and the schedules. No file is over 5 MB. No simulator code changed, so `SIM_CODE_VERSION` and
the references are as they were.

**Which seasons load.** Without an F1TV login, `livetiming.formula1.com` now answers 401 for
every season. FastF1 3.9 then falls back to its own mirror, `livetiming-mirror.fastf1.dev`,
which serves 2021 and 2022 but answers 404 for 2018-2020 (checked on 2018 Bahrain, 2019
Bahrain and Abu Dhabi, and 2020 Spain and Emilia Romagna). 2023 on needs the login. No login
was used, so only 2021-22 could be added. The 2019 and 2020 races already in the cache were
fetched before the login requirement.

**Group A: history at circuits already in use.** The list asked for Monaco 2018; Bahrain 2018,
2019, 2021 and 2022; Austria 2018, 2019, 2021, 2022 and 2023; Barcelona 2018, 2019, 2020 and
2022; Silverstone 2018 and 2020; Jeddah 2021; Imola 2020, 2021 and 2022; and Lusail 2021 and
2023. Of these, the nine 2021-22 weekends are cached. The other eleven (2018-2020 and 2023)
could not be fetched. 2021 Emilia Romagna has no FP1 on the mirror, so FP2, FP3, Qualifying
and Race are cached.

**Group B: a seeded pool.** The pool is every round of 2021-22 (the seasons that load) that
was neither cached nor in group A, in season and round order: 33 races. k = 15 fills the
budget of about 24 new weekends.

```python
pool = ["2021_03_portugal", "2021_05_monaco", "2021_06_azerbaijan", "2021_07_france", "2021_08_styrian",
        "2021_10_united_kingdom", "2021_11_hungary", "2021_12_belgium", "2021_13_netherlands", "2021_14_italian",
        "2021_15_russia", "2021_16_turkey", "2021_17_united_states", "2021_18_mexico", "2021_19_brazil",
        "2021_22_united_arab_emirates", "2022_03_australia", "2022_05_miami", "2022_07_monaco",
        "2022_08_azerbaijan", "2022_09_canada", "2022_10_united_kingdom", "2022_12_france", "2022_13_hungary",
        "2022_14_belgium", "2022_15_netherlands", "2022_16_italian", "2022_17_singapore", "2022_18_japan",
        "2022_19_united_states", "2022_20_mexico", "2022_21_brazil", "2022_22_united_arab_emirates"]
random.seed(2026); random.sample(pool, 15)
# ['2021_12_belgium', '2022_09_canada', '2022_03_australia', '2022_20_mexico', '2022_21_brazil',
#  '2022_17_singapore', '2021_07_france', '2022_22_united_arab_emirates', '2022_08_azerbaijan',
#  '2022_14_belgium', '2022_05_miami', '2021_18_mexico', '2022_07_monaco', '2022_12_france',
#  '2021_22_united_arab_emirates']
```

**Eligibility facts.** These come from `calibration.eligibility` on each race's lap frame and
race control messages, with no simulator fit. The total laps are the session's. A "None" or
"UNKNOWN" compound is a lap with no compound label, which the eligibility rule counts as a
slick. The new races were not looked at beyond these facts: no fit, calibration, cross-race
pass, scenario, reference, baseline or agent run.

| race id | group | format | laps | eligible drivers | dry compounds | eligible, or why not | MB |
|---|---|---|---|---|---|---|---|
| 2021_01_bahrain | A | conventional | 56 | 16 | H M S | eligible | 1.76 |
| 2021_02_emilia_romagna | A | conventional, no FP1 | 63 | 17 | M S | no: wet (INTERMEDIATE, WET), red flag | 1.59 |
| 2021_09_austrian | A | conventional | 71 | 19 | H M S | eligible | 1.92 |
| 2021_20_qatar | A | conventional | 57 | 18 | H M S | eligible | 1.71 |
| 2021_21_saudi_arabia | A | conventional | 50 | 15 | H M S | no: red flag | 1.50 |
| 2022_01_bahrain | A | conventional | 57 | 18 | H M S | eligible | 1.87 |
| 2022_04_emilia_romagna | A | sprint | 63 | 18 | H M S | no: wet (INTERMEDIATE) | 1.97 |
| 2022_06_spain | A | conventional | 66 | 18 | H M S | eligible | 1.72 |
| 2022_11_austria | A | sprint | 71 | 17 | H M (and None) | eligible | 2.13 |
| 2021_07_france | B | conventional | 53 | 20 | H M S | eligible | 1.74 |
| 2021_12_belgium | B | conventional | 39 | 0 | UNKNOWN | no: red flag (lap 1), 0 eligible drivers, 1 dry compound | 0.62 |
| 2021_18_mexico | B | conventional | 71 | 17 | H M S | eligible | 1.78 |
| 2021_22_united_arab_emirates | B | conventional | 58 | 14 | H M S | eligible | 1.68 |
| 2022_03_australia | B | conventional | 58 | 17 | H M S | eligible | 1.61 |
| 2022_05_miami | B | conventional | 57 | 16 | H M S | eligible | 1.74 |
| 2022_07_monaco | B | conventional | 77 | 0 | H M S (and None) | no: wet (INTERMEDIATE, WET), red flag (lap 1), 0 eligible drivers | 1.88 |
| 2022_08_azerbaijan | B | conventional | 51 | 15 | H M S | eligible | 1.56 |
| 2022_09_canada | B | conventional | 70 | 17 | H M | eligible | 1.75 |
| 2022_12_france | B | conventional | 53 | 15 | H M | eligible | 1.63 |
| 2022_14_belgium | B | conventional | 44 | 18 | H M S | eligible | 1.42 |
| 2022_17_singapore | B | conventional | 61 | 12 | M S | no: wet (INTERMEDIATE) | 1.60 |
| 2022_20_mexico | B | conventional | 71 | 18 | H M S | eligible | 1.88 |
| 2022_21_brazil | B | sprint | 71 | 17 | H M S | eligible | 2.16 |
| 2022_22_united_arab_emirates | B | conventional | 58 | 17 | H M S | eligible | 1.76 |

Of the 24 races, 18 are eligible: 6 in group A and 12 in group B. The ineligible ones still
give practice and partial history.

**Fresh held-out draw (proposed here, registered in `data/eval_set.json` on 2026-11-21; see
"The registered split").** The final test set combines a fresh sample of these races with the
held-out races already registered. Results can then be shown across regulation eras: 2021 is
the previous car generation, 2022 the first ground-effect season, and 2024-25 later
ground-effect seasons. The draw is stratified by season, from the eligible group-B
races only (group A stays calibration and history), in season and round order within each
season, with one seed. A season with fewer than 5 eligible group-B races gives 2 instead of
3. 2021 has 3, so it gives 2:

```python
pool_2021 = ["2021_07_france", "2021_18_mexico", "2021_22_united_arab_emirates"]
pool_2022 = ["2022_03_australia", "2022_05_miami", "2022_08_azerbaijan", "2022_09_canada", "2022_12_france",
             "2022_14_belgium", "2022_20_mexico", "2022_21_brazil", "2022_22_united_arab_emirates"]
random.seed(2027)
random.sample(pool_2021, 2)   # ['2021_07_france', '2021_18_mexico']
random.sample(pool_2022, 3)   # ['2022_05_miami', '2022_03_australia', '2022_14_belgium']
```

All five drawn races are conventional weekends. The only eligible sprint weekend in the pools
was 2022 Brazil, and it was not drawn.

The other 13 new eligible races are calibration data. From group A: 2021_01_bahrain,
2021_09_austrian, 2021_20_qatar, 2022_01_bahrain, 2022_06_spain and 2022_11_austria. From
group B: 2021_22_united_arab_emirates, 2022_08_azerbaijan, 2022_09_canada, 2022_12_france,
2022_20_mexico, 2022_21_brazil and 2022_22_united_arab_emirates. Now that the five are in
`data/eval_set.json`, `--races auto` leaves them out.

`jobs.load_race` builds 2022_11_austria: its 299 laps with no recorded compound (whole stints of
twelve cars, Compound None in the timing data) are labelled `UNKNOWN` by `data.lap_frame`, with
a log warning per load. They keep their timing (other cars' gaps and positions need it) but are
never clean, so no fit reads them; `calibration.run` does not replay a driver with one, and
`snapshot_as_of` builds no snapshot on one (the sampler skips the point with that reason).

## Rerunning the evaluation

`python -m src.eval.rerun <stage> --set <name> --world-config <file|default> --belief-config
<file|default> [--jobs N]` reruns the whole evaluation under the corrected protocol, one command
per stage, under `--out-dir` (default `reports/eval/rerun/<set>`). The sets are in
`configs/eval/rerun_sets.yaml`:

- `dev20`: the nineteen calibration races that are ever sampled (the six development races and
  thirteen 2021-22 weekends; 2025 Qatar, the twentieth calibration race, is calibration only), 25
  scenarios per race, seed 0, a development set.
- `test9`: the nine races of `data/eval_set.json`, 25 per race, seed 0, a report set, tables split
  by era (2019-20, 2021, 2022, 2024-25). It is built once, later, on the owner's word: the
  `scenarios` stage refuses a report set without `--final`, because building it is the first load
  of the five new held-out races.

| stage | what it does |
| --- | --- |
| `scenarios` | the set with the leak-free sampler (`--rule as_of`, the default, or `stratified`), builder `calibration.snapshot_as_of`; never over an existing `scenarios.json` |
| `references` | every scenario's reference on the world config, race-parallel (`--jobs`, spawned workers, BLAS at one thread each) and resumable (`parts/references/<race>.pkl`); the cache under `references/` |
| `baselines` | every non-LLM policy (below) on the belief config, race-parallel and resumable (`parts/baselines/<race>.csv`), the references required; `baselines.csv` |
| `agent` | one sequential pass of the live agent (`LLM_BACKEND`), its tools on the belief config (`AGENT_BELIEF_CONFIG`), trajectories to `trajectories/`; refuses to overwrite `agent.csv` or a non-empty `trajectories/` |
| `tables` | `tables/`: the combined rows, per-race and pooled summary, agent-minus-X paired tables with (race, driver) cluster intervals for every comparator (belief_optimizer stands in when there are no agent rows, named in the file), the chart CSV, the tail table (share over 10, 20, 30 s, median, p90, worst), status counts by reason, by era for a set with eras, position metrics when traffic is on; and `provenance.json` |

`provenance.json` records the commit (and tag, and whether the tree was dirty), the world and
belief configs with their file sha256 and config_hash, SIM_CODE_VERSION, the set's digest,
builder and sampling rule, the profile, the model and loop settings from the environment, the
stamps the rows carry, the future-blindness harness on this commit (the fixed and the still
known leaks of `tests/future_blindness.py`; with `--harness` the default part's counts), the as-of
tables (`src/sim/provenance.py`: each artifact's kind, file hashes, the races it records and, per
race of the set, the races it read), and every command run in the directory (`commands.log`).

What each non-LLM policy does now (the stint limit is retired: R8 off, the wear cliff on; a plan
is invalid only for a breach of R1-R4, the published regulations, or a compound the simulator
cannot price):

- `fixed_one_stop`: one stop at half the remaining laps (kept inside the minimum stints) onto the
  hardest other dry compound; when the current set would run past the belief's as-of tyre-life
  estimate (`Simulator.cliffs_for`: the larger of the longest run seen so far and the pre-race prior
  from earlier races, `tyre_history.tyre_life_prior`), the stop moves earlier to the last lap that
  keeps the plan legal and inside it.
- `fixed_two_stop`: two stops splitting the remaining laps in thirds, current set, the other
  compound (MEDIUM, HARD, SOFT preference), the current compound again; not tyre-life-aware.
- `greedy_crossover`: pits when the current set's next lap on the belief's pace is slower than the
  fastest new set's first lap (no pit-loss term, at most the future-stop limit); a lap past the
  as-of tyre-life estimate costs `PAST_LIMIT_S`, so it stops there when a stop is left.
- `actual_strategy`: what the team did, its last set run to the scheduled distance (a comparator; it
  reads the future by definition).
- `run_to_flag`: no further stop when that is legal and inside the as-of tyre-life estimate, else
  the fewest stops (each as late as possible, softest compound first) that are.
- `one_stop_total`: `fixed_one_stop` before the driver's first stop, `run_to_flag` after it.
- `belief_optimizer`: the optimizer on the belief (`decisions.belief_optimum`), its own plan.

## Metrics

For each scenario the optimizer (`optimizer_bridge.optimize_batch`, Quan's search scored by
Jason's simulator) finds the world's best plan with at most `max_stops` future stops. Every
policy's plan is scored by the world under the same draws (docs/rules.md R6).

| column | meaning |
| --- | --- |
| `regret_s` | mean remaining time of the policy's plan minus the optimizer's. A plan with more stops than the optimizer searches can come in below it, so this can be negative. |
| `regret_pct` | `regret_s` as a percentage of the optimizer's remaining time |
| `invalid_rate` | invalid plans / (ok + invalid). Invalid: a rule broken (docs/rules.md) or no plan submitted. |
| `stop_agreement` | share of scored plans with the optimizer's number of future stops |
| `first_pit_error` | \|first pit lap - optimizer's first pit lap\|; 0 when neither stops, blank when only one does |
| `latency_s`, `tool_calls`, `input_tokens`, `output_tokens` | per decision, when the policy reports them. `tool_calls` counts every call the agent attempted, refused calls, repeats and submission attempts included |
| `evidence_calls` | agent rows: the evidence calls that ran (no submissions, no refused calls, no repeats served from the cache; a repeat run again counts); summary `mean_evidence_calls` beside `mean_tool_calls` |
| `repeat_calls` | agent rows: the evidence calls identical to an earlier one in the episode, run again (`AGENT_REPEAT_MODE=run`, the default) or answered with its cached result and no budget (`cache`); blank for other policies and for rows from before this column; summary `mean_repeat_calls` |
| `pit_now_opt` | the world optimizer's first stop is at the end of this lap (`opt_first_pit_lap == laps_completed + 1`) |
| `submitted_in_grace` | agent rows: the submission came in a grace turn, after the evidence turns ran out (blank for other policies) |
| `decision`, `p_pit` | agent rows only: PIT or STAY_OUT, and the stated probability that pitting now is right (`confidence` for PIT, `1 - confidence` for STAY_OUT) |
| `brier` | summary, agent rows: mean of `(p_pit - pit_now_opt)^2`. Only in tables of rows that have `p_pit`. |
| `pit_now_agreement` | summary, agent rows: share of decisions where PIT matches `pit_now_opt` |
| `final_position_mean`, `opt_final_position_mean` | world traffic on (the default): the plan's and the optimizer's mean finishing position over the samples, 1 + the ghosts that crossed the line at the end of the last lap before the plan's car, on the traffic model's own clock (`BatchResult.final_pos`: the driver's own pace on the track and the neutral-lap alignment, "Traffic" below); a DNF sample is behind every finisher, ordered by `dnf_lap` against the ghosts that retired. Information only: the headline metric stays `regret_s` |
| `position_regret` | `final_position_mean - opt_final_position_mean`; positive means finishing behind the reference. NaN with world traffic off or when traffic cannot be placed (no `elapsed_s`): without ghosts there is no field to finish in |
| `mean_position_regret`, `mean_final_position` | summary, only when some row has a position |
| `ref_models_agree` | optional (`--world-configs`): every alternative world's reference has the same number of stops and a first pit lap within 2 of the world's. `score` then also writes `*_summary_agree.csv`, the table over the agreeing rows |
| `regret_se_s` | the paired standard error of `regret_s` under the shared draws: `sd(policy_times - reference_times) / sqrt(N)`, sd with N - 1, over the N per-draw remaining times (the reference's in float32, from the reference cache). 0 at N = 1 |
| `within_mc` | `\|regret_s\| <= 2 * regret_se_s`: at this N the draws cannot tell the plan from the reference. At N = 1 (or in a world without drawn events, where every draw is the same) the standard error is 0, to rounding, and this is the tie rule, `regret_s` 0 |
| `share_within_mc` | summary: share of scored rows with `within_mc` |
| `opt_set_size` | near ties (information only): the searched plans whose mean is within 2 paired standard errors of the reference's (`gap <= 2 * SE`, the reference included), the optimal set. NaN from a format-1 reference cache |
| `best_pit_now_s`, `best_stay_s` | near ties: the mean of the best searched plan that pits at the end of this lap, and of the best one that does not; NaN where there is no such plan (pitting now would leave a stint under the minimum, say) |
| `pit_now_gap_s`, `pit_now_gap_se_s` | near ties: `best_pit_now_s - best_stay_s` and the paired standard error of that difference over the shared draws |
| `pit_now_decisive` | near ties: `\|pit_now_gap_s\| > 2 * pit_now_gap_se_s`; empty where one of the two plans does not exist |
| `stop_agree_set`, `first_pit_error_set` | near ties, ok rows: `stop_agree` and `first_pit_error` against the closest member of the optimal set (some member makes the same number of stops; the smallest first pit error over the members) |

Row statuses: `ok`, `invalid`, `unavailable` (the policy does not apply here, e.g. the actual
strategy of a driver whose real stint broke the minimum), `error` (the policy itself failed,
e.g. an API error; not counted as invalid), `no_reference` (no legal plan exists).

Each row is stamped with the scenario set version and digest, `belief_config_hash`,
`world_config_hash`, `belief_window`, `sim_code_version`, the seed and `n_samples` (and
`ref_world_config_hashes` with `--world-configs`). Rows scored before the split carry one
`config_hash` instead. Rows also carry the world's R8 settings, `stint_limit_rank` and
`stint_limit_margin_laps` (blank: R8 off), and `summarize` copies both into every summary
row (`score.SUMMARY_STAMPS`; values joined by `;` if rows of two settings are summarized
together), so a rank-1 and a rank-3 table can be shown side by side. Files scored before
the stamps have neither column.

## Reference cache

The reference of a scenario (the world optimizer's plan and its per-draw times) depends only
on the scenario set, the world config, the simulator code, the seed, `n_samples`,
`max_stops` and whether the events are fixed to the record, so it is computed once
(`src/eval/reference.py`):

```bash
python -m src.eval reference --scenarios data/scenarios/v1.json [--world-config W] [--profile P] \
    [--hindsight] [--seed 0] [--n-samples N] [--max-stops K]
python -m src.eval reference --check        # cached references from another sim_code_version
```

**Expected exception: d3.** d3 is rebuilt at the final evaluation. Until then its cached
reference (`data/references/d3/9f933a611bea.parquet`, sim_code_version `b62094155bde`) is
stale on purpose and is neither rebuilt nor rescored. `reference --check` lists it apart,
under "expected stale (docs/eval.md)" with that reason, and does not count it as stale
(`reference.EXPECTED_STALE`). `stale_references` still returns it, so a real run's
references stage rebuilds it when that run is made.

It writes `data/references/<set version>/<key>.parquet`, with the inputs in `<key>.json`
beside it, where `key = sha256(set digest, world config hash, sim_code_version, seed,
n_samples, max_stops, hindsight)[:12]` (`reference.reference_key`; `--n-samples` and
`--max-stops` default to the profile's, as in `score`). Per scenario the table holds the
reference plan, its per-draw remaining times (float32), their mean, sd and `cvar_90`
(mean of the worst 10 percent), `pit_now_opt`, the runner-up's mean (the next plan in the
optimizer's tie-break order) and the paired standard error between the two. Since format 2
(`reference.REFERENCE_FORMAT`, `"format"` in the sidecar) it also holds, in tie-break order,
the searched plans whose gap to the reference is at most max(10 s, 5 x their paired SE)
(`STORE_GAP_S`, `STORE_SE`), plus the best pit-now and best stay-out plans always, so
neither the optimal set (gap <= 2 SE, inside that bound) nor `pit_now_gap_s` depends on the
bound. For each: its mean minus the reference's (`plan_gap_s`) and
its paired standard error against the reference (`plan_se_s`), both float32, its number of
stops and first pit lap, and the paired standard error of the best pit-now plan against the
best stay-out plan. Those give the near-tie columns. A format-1 cache still loads (its
near-tie columns are NaN) and `reference --check` lists it as stale. The command
prints the reference's precision: how far the runner-up is above the reference and whether
the gap is within 2 standard errors.

`score` looks up the reference for its own inputs under `--reference-dir` (default
`data/references`). A hit is used as it is; a miss computes the reference the same way and
prints `reference computed, not cached` once per run. Cached and computed references give
identical rows. With `--require-reference` a miss is an error: report runs use it. The
`jobs` runner reads the cache the same way (`python -m src.eval jobs --reference-dir D
--require-reference ...`); both travel in the runner spec, so on Sol a report run's
workers fail their first chunk on a miss instead of computing. `--hindsight` fixes every draw's SC and VSC laps to
the ones the driver recorded over the remaining laps; `score` never reads a hindsight
reference, since a regret needs both plans under the same draws.

The key carries `sim_code_version`, a physics version: a hash of the modules listed in
`calibration.PHYSICS_MODULES` (engine, degradation, events, hazard, track, traffic, rules,
config, data, tyre_sets, weather, pace, calibrators, optimizer_bridge, registry, event_rates,
fuel_prior, proxy) plus `src/contracts.py`. Every other module of `src/sim` is on
`calibration.NON_PHYSICS` (the package `__init__`, agent_tools, jobs, compute, calibrate,
calibration, model_report, race_ids, cross_race, real_run, compound_steps), and a test fails on a module on neither list, so a new module is classified on
purpose. calibration is non-physics because the scenario sets it builds are frozen by their
digest. A change to a physics module orphans the cache: `reference --check` lists what is
stale, and `reference` rebuilds it; a change to the tooling (agent tools, jobs) does not.
Moving to the allowlist moved every `sim_code_version` stamp once; the committed reports
keep their old stamps (stamp-only drift, reports/calibration/README.md).
`optimizer_bridge.optimize_batch(..., return_times=True)` (off by default) is how the cache
gets the per-draw times: `diagnostics["times"]`, `diagnostics["runner_up"]` and
`diagnostics["searched"]` (every legal plan in tie-break order with its mean and times).

Committed (SPEC 0023, the R8 stint limit at rank 1, local profile: seed 0, 50 samples, two
stops, default world with the compound proxy's blend), at sim_code_version `58b50a00e0e3`:
`data/references/v1/c0c7247b95b7.parquet`, `data/references/d2/dbc5b90ac885.parquet`,
`data/references/pilot1/fda18509e2f7.parquet` and `data/references/dev6/fbb33ef732dd.parquet`.
They replace `820f58b3bce2`, `763a6d2cfb45`, `c87fdfd26007` and `4ecc62a30937` (sim_code_version
`1dcf46ad75aa`), on which the committed `*_r8` and dev6 result files were scored; what the
proxy changed in them is in "Proxied compounds". With the proxy off, every record of the four
sets is identical to those (rebuilt in a scratch directory and compared column by column).

**Rebuilt at the current simulator.** The first R8 references (`441a798fbebb`, `182013ff1f42`,
`a4241208bfb7`, at sim_code_version `983fd6f4f2bf`, built after main was merged in for #61)
went stale when #69 (the `_ParameterDraws._plan` fix in degradation.py), #60 and the driver
effects code (off by default) moved the physics version. They were rebuilt once with the
sidecars' inputs and the stale files removed. The new parquet files are byte-identical to the
ones they replace (every scenario's plan, per-draw times and near-tie columns), so under the
default world none of those changes moves a v1, pilot1 or d2 reference, and the `*_r8` result
files below, scored on the old keys, hold for the new ones. `reference --check` now lists only
d3, the expected exception.

v1 scored on these references (baselines with the limit-aware wrapper, and the recorded run):
`reports/eval/v1_baselines_r8.csv`; pilot1 and d2 baselines: `pilot1_baselines_r8.csv` and
`d2_baselines_r8.csv` ("Result files" below).

The record below is of the reference before R8.

Before: `data/references/v1/396dff32d244.parquet` (135 KB, format 2; the format-1 file
was 16 KB, and keeping every searched plan would be 595 KB), v1 under the default world (local profile: seed 0, 50 samples, two stops) at
sim_code_version `21b81a57ea42`. From
it, `score --require-reference` reproduced `reports/eval/v1_baselines_split.csv` row for
row (every value column; the stamp columns `latency_s`, `sim_code_version`,
`belief_config_hash` and `world_config_hash` are excluded, stamp-only drift under
reports/calibration/README.md); the rows gain `regret_se_s` and
`within_mc`. That world draws no events, so every `regret_se_s` is 0 and `within_mc` is
the tie rule there. The rows also gain the near-tie columns, and in that world every
optimal set is the reference alone (`opt_set_size` 1 on all 1320 rows), every pit-now
call with both kinds of plan is decisive, and the set-based agreement equals the plain one.

Under R8 at the defaults (`reports/eval/v1_baselines_r8.csv`) that no longer holds: 6 of the
264 v1 scenarios (30 of 1320 rows), all HAM, laps 0, 10, 11, 15, 20 and 25, have exact ties,
`opt_set_size` 2 or 4. The references there are two-stops, and in this deterministic linear
world two kinds of plan cost exactly the same: the later stints swapped (S22 M17 S27 against
S22 S27 M17), and one SOFT lap moved between two SOFT stints that end at the same tyre age
(from lap 0 on a 4-lap-old set, S22 M17 S27 against S23 M17 S26). R7 picks one: equal stop
counts, then the earlier pit laps, so S22 M17 S27 (stops after laps 22 and 39). Every pit-now
call is still decisive and the set-based stop agreement equals the plain one; the set-based
first pit error is one lap lower on 8 rows.

### Near ties (information only)

`python -m src.eval report` prints, below the existing tables, a near-tie view when the rows
carry the near-tie columns (`score.summarize_near_ties`), one row per policy over every
race: the share of scenarios whose optimal set has more than one plan, the share where the
pit-now call is decisive (of the scenarios with both a pit-now and a stay-out plan), and
the stop agreement and mean first pit error against the reference and against the closest
member of the optimal set. Nothing in the summary table or the headline metrics changes.
Under the deterministic default world the view shows every call decisive and sets of size
1, except the exact ties R8 brings (above: 6 v1 scenarios with sets of 2 or 4).

v1 under the default world with `events: {name: rate}` (default rates), 200 samples, belief
default, baselines and the recorded run, from a cached reference:

| policy | scenarios | share_opt_set_gt1 | share_pit_now_decisive | scored | stop_agreement | stop_agreement_set | mean_first_pit_error | mean_first_pit_error_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actual_strategy | 264 | 0.78 | 0.84 | 199 | 0.59 | 0.60 | 9.99 | 8.22 |
| baseline_agent_recorded | 264 | 0.78 | 0.84 | 1 | 0.00 | 0.00 | 14.00 | 10.00 |
| fixed_one_stop | 264 | 0.78 | 0.84 | 250 | 0.86 | 0.86 | 8.14 | 6.12 |
| fixed_two_stop | 264 | 0.78 | 0.84 | 211 | 0.05 | 0.05 | 9.28 | 7.03 |
| greedy_crossover | 264 | 0.78 | 0.84 | 185 | 0.64 | 0.64 | 15.54 | 13.31 |

Per scenario the optimal set has median 5 plans (max 9; 58 of 264 have one). 138 scenarios
have both a pit-now and a stay-out plan; on 116 the call is decisive (median |gap| 1.69 s
against a median paired SE of 0.24 s), and of the 22 that are not, 13 have the reference
pitting now. The near ties sit within one strategy: the set-based stop agreement moves by
at most 0.01, while the first pit error drops by about 2 laps for every policy.

### Precision under drawn events (v1, N = 200)

The default world with `events: {name: rate}` (default rates), 200 samples, belief
default, baselines and the recorded run, from a cached reference:

| policy | scored | mean regret s | share_within_mc | mean regret_se_s |
| --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.90 | 0.17 | 0.27 |
| actual_strategy | 199 | 6.52 | 0.22 | 0.24 |
| greedy_crossover | 185 | 11.69 | 0.04 | 0.29 |
| fixed_two_stop | 211 | 12.85 | 0.00 | 0.33 |
| baseline agent (recorded) | 1 | 7.33 | 0.00 | 0.39 |

Reference against runner-up on the 250 scenarios with more than one legal plan: paired
standard error mean 0.19 s (median 0.19, max 0.29); the runner-up's gap has median 0.07 s
(mean 1.45), and on 82% of them it is within 2 standard errors. At 200 draws the reference
is not told apart from the next plan on most scenarios, while regrets of a few seconds are
well outside the noise (a regret_se_s of about 0.3 s).

## physics-v1

`physics-v1` is the simulator release the report numbers are computed against: a git tag
at the end of sprint 2, with the sim_code_version below. Report runs score with
`--require-reference` against references cached at that sim_code_version, so a report
number cannot come from another simulator version without failing.
`data/ground_truth.json` stays the frozen legacy check (`legacy_tools.optimal_strategy`,
`tests/test_case.py`) and is not regenerated from this simulator.

| tag | sim_code_version | commit |
| --- | --- | --- |
| physics-v1 | tagged at the end of sprint 2 (Nov 20); its SIM_CODE_VERSION is recorded here when the tag is made | not tagged yet |

TODO (end of sprint 2, the user): apply the tag with `git tag physics-v1 <commit>` on the
commit that freezes the simulator, put the commit here, and record the sim_code_version
above from `python -c "from src.sim.calibration import SIM_CODE_VERSION as v; print(v)"` at
that commit. Then
rebuild the report references (`python -m src.eval reference ...`) and check
`python -m src.eval reference --check` lists nothing.

## Result files

Which committed table is current. Every current file has its `*_summary.csv` beside it.

**Superseded by the tyre-life change.** Every result below that was scored before it was
produced under the hindsight stint limit R8 (docs/rules.md R8, retired) and without the wear
cliff, so all of them are superseded: the held-out d4 result at tag `d4-final-eval`
(`d4_baselines.csv`, `d4_agent.csv`, `d4_combined*.csv`, `d4_rule_baselines.csv` and their
compare tables), the dev6 live agent runs 1-3 (`dev6_agent_pass1*`, `dev6_run2_*`,
`dev6_run3_*`), the rule-baseline files (`dev6_rule_baselines.csv`, `d4_rule_baselines.csv`,
`*_combined_rules.csv`, `*_rules_compare_*`), the dev6 baselines (`dev6_baselines.csv`,
`dev6_run2_baselines.csv`) and the v1, pilot1 and d2 `*_r8` tables. They stay as the record of
that setting; `configs/sim/legacy/world_r8.yaml` with `configs/sim/legacy/belief_r8.yaml`
reproduces it. The current development table is the first row. d4 is re-evaluated in a later,
separate step (its reference is on the expected-stale list); until then no held-out number is
current.

| set | current | superseded (kept as the record) |
| --- | --- | --- |
| dev6, non-LLM policies under the tyre-life physics (development, a sanity check, not a result to tune against) | `reports/eval/dev6_tyrelife_baselines.csv` (baselines, `belief_optimizer`, `run_to_flag`, `one_stop_total`; world `config.WORLD_DEGRADATION` with the cliff, R8 off; reference rebuilt on this branch) | every R8-era dev6 table below |
| v1, baselines + recorded run | `reports/eval/v1_baselines_r8.csv` (R8 rank 1, margin 3, limit-aware baselines) | `v1_baselines_split.csv` (Milestone 2: belief/world split, before R8); `v1_baselines.csv` (before the split) |
| v1, world traffic on | none under R8 yet | `v1_world_baselines.csv` (before R8) |
| pilot1, baselines | `reports/eval/pilot1_baselines_r8.csv` | `pilot1.csv` baseline rows (Milestone 2, before R8) |
| pilot1, live agent | `pilot1.csv` agent rows (Milestone 2, before R8; the cited ones) | `pilot1_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d2, baselines | `reports/eval/d2_baselines_r8.csv` | `d2.csv` baseline rows (Milestone 2, before R8) |
| d2, live agent | `d2.csv` agent rows (Milestone 2, before R8; the cited ones) | `d2_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d3 | none: its Monaco rows are superseded by d4 (below); d3 is not rescored | `d3_baselines.csv` (Nov 11, before R8; its reference is the expected stale one) |
| d4, baselines + belief_optimizer (**report set, the held-out result**) | `reports/eval/d4_baselines.csv` (reference `data/references/d4/39bf1bbb4aa9.parquet`) | supersedes d3's 2019 Monaco baseline rows |
| d4, live agent, one pass (**report set, the held-out result**) | `reports/eval/d4_agent.csv`, trajectories in `reports/eval/trajectories/d4/`; `d4_combined.csv` (agent + baseline + belief_optimizer rows, the input to `d4_compare_regret_s_*` with driver clusters and `d4_compare_race_regret_s_*` with race clusters); per-race paired table `d4_compare_per_race_regret_s_paired.csv`; figure table `d4_chart.csv` | |
| dev6, baselines (development, calibration races) | `reports/eval/dev6_baselines.csv` (R8 rank 1, margin 3, limit-aware baselines) | |
| dev6, live agent pass 1 (development, calibration races) | `reports/eval/dev6_agent_pass1.csv`, trajectories in `reports/eval/trajectories/dev6_pass1/`; `dev6_pass1_combined.csv` (agent + baseline rows, the input to `dev6_pass1_compare_regret_s_*` with driver clusters and `dev6_pass1_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_pass1_compare_per_race_regret_s_paired.csv`; figure table `dev6_pass1_chart.csv` | |
| dev6, baselines + belief_optimizer, run 2 (development, calibration races) | `reports/eval/dev6_run2_baselines.csv` (compound proxy on, reference `fbb33ef732dd`) | `dev6_baselines.csv` (before the proxy) is the record pass 1 was compared with |
| d4, rule baselines, scored once after the d4 results (**report set**) | `reports/eval/d4_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, frozen at `5a2bc41`); `d4_combined_rules.csv` (`d4_combined.csv`'s rows plus these) and `d4_rules_compare_*` ("Rule baselines added after the held-out run") | `d4_combined.csv` stays as committed |
| dev6, rule baselines (development, calibration races) | `reports/eval/dev6_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, reference `fbb33ef732dd`); `dev6_run2_combined_rules.csv` (run 2's rows plus these) and `dev6_run2_rules_compare_*` ("Rule baselines added after the held-out run") | |
| dev6, live agent run 2 (development, calibration races) | `reports/eval/dev6_run2_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run2/`; `dev6_run2_combined.csv` (agent + baseline + belief_optimizer rows, the input to `dev6_run2_compare_regret_s_*` with driver clusters and `dev6_run2_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run2_compare_per_race_regret_s_paired.csv`; figure table `dev6_run2_chart.csv` | pass 1 stays as the record of the earlier loop and world |
| dev6, live agent run 3 (variance repeat of run 2; development, calibration races) | `reports/eval/dev6_run3_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run3/`; `dev6_run3_combined.csv` (run 3's agent rows + run 2's baseline and belief_optimizer rows, the input to `dev6_run3_compare_regret_s_*` with driver clusters and `dev6_run3_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run3_compare_per_race_regret_s_paired.csv`; figure table `dev6_run3_chart.csv` | not a replacement: run 2 stays the cited dev6 result; run 3 only measures run-to-run variation |
| dev6, live agent run 4 (loop that logs and recovers from turns with no usable tool call; development, calibration races) | `reports/eval/dev6_run4_agent.csv`, trajectories (with `no_tool_call` events and `parse_repair` notes) in `reports/eval/trajectories/dev6_run4/`; `dev6_run4_combined.csv` (run 4's agent rows + every non-agent row of `dev6_run2_combined_rules.csv`, rule baselines included, the input to `dev6_run4_compare_regret_s_*` with driver clusters and `dev6_run4_compare_race_regret_s_*` with race clusters) | a different agent loop from runs 2 and 3 and from the held-out d4 pass, which all ran before the fix |

Rows scored under R8 carry `stint_limit_rank` and `stint_limit_margin_laps`; rows scored after
the tyre-life change carry a blank `stint_limit_margin_laps` (R8 off).

## Current numbers (v1, R8, deterministic)

`reports/eval/v1_baselines_r8.csv`: `python -m src.eval score --scenarios data/scenarios/v1.json
--policies baselines --recorded data/eval/baseline_agent_sample_run.json --require-reference
--out reports/eval/v1_baselines_r8.csv`, at R8 rank 1 and margin 3, with the limit-aware
baselines, against the Milestone 2 table (`v1_baselines_split.csv`, before R8). Overall:

| policy | scored / invalid / unavailable, M2 | R8 | mean regret s, M2 | R8 | median, M2 | R8 | p90, M2 | R8 | stop agreement, M2 | R8 | first-pit error, M2 | R8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 / 0 / 14 | 250 / 0 / 14 | 4.95 | 4.33 | 3.01 | 2.14 | 13.60 | 13.45 | 0.90 | 0.86 | 6.80 | 4.67 |
| actual_strategy | 199 / 0 / 65 | 199 / 0 / 65 | 6.72 | 5.49 | 7.11 | 5.33 | 14.52 | 12.06 | 0.57 | 0.61 | 9.98 | 6.66 |
| fixed_two_stop | 211 / 0 / 53 | 197 / 14 / 53 | 12.95 | 11.55 | 9.25 | 8.46 | 21.18 | 21.18 | 0.00 | 0.03 | 8.77 | 4.96 |
| greedy_crossover | 185 / 0 / 79 | 165 / 20 / 79 | 11.91 | 8.97 | 11.54 | 8.40 | 20.14 | 20.14 | 0.70 | 0.15 | 13.73 | 2.77 |
| baseline agent (recorded) | 1 / 0 / 263 | 1 / 0 / 263 | 6.29 | 2.05 | | | | | 0.00 | 1.00 | 13.00 | 6.00 |

- The unavailable rows are exactly Milestone 2's (a zero-lap first stint, a real stint
  under v1's 10-lap minimum, no recorded decision); R8 makes none. The rows the first
  version of #61 marked unavailable for R8 now come out as follows. fixed_one_stop: 91 of 91
  scored. greedy_crossover: 109 of 129 scored and 20 invalid (past the limit with no stop
  left). fixed_two_stop: 14 invalid (not limit-aware).
- HAM lap 0: fixed_one_stop's S33 > M33 ends the 4-lap-old SOFT at 37, past 35; the wrapper
  gives S31 > M35 (regret 0.40 s). greedy_crossover's S10 > M10 > S46 becomes
  S4 > M3 > S59, still past the limit with no stop left: invalid.
- greedy_crossover's stop agreement falls (0.70 to 0.15) because the forced stop is
  usually followed by a second crossover stop, a two-stop against mostly one-stop references.
- No row has negative regret.

On the rows scored ok both times, Milestone 2 being `v1_baselines_split.csv` (not the older
`v1_baselines.csv`) (mean of R8 minus M2, 95% cluster-bootstrap interval over
(race, driver), `eval.compare.cluster_bootstrap`, 2000 draws):

| policy | n | mean regret M2 | R8 | R8 - M2 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.95 | 4.33 | -0.63 | [-0.84, -0.51] |
| actual_strategy | 199 | 6.72 | 5.49 | -1.23 | [-1.58, -0.85] |
| fixed_two_stop | 197 | 13.19 | 11.55 | -1.64 | [-1.95, -1.34] |
| greedy_crossover | 165 | 11.47 | 8.97 | -2.51 | [-3.93, -0.95] |
| baseline agent (recorded) | 1 | 6.29 | 2.05 | -4.24 | |

pilot1 and d2 baselines under R8 (`pilot1_baselines_r8.csv`, `d2_baselines_r8.csv`) against the
baseline rows of their Milestone 2 files, scored / invalid / unavailable and mean regret s:

| set | policy | M2 | R8 | mean regret M2 | R8 |
| --- | --- | --- | --- | --- | --- |
| pilot1 | fixed_one_stop | 25 / 0 / 0 | 25 / 0 / 0 | 5.28 | 4.66 |
| pilot1 | actual_strategy | 25 / 0 / 0 | 25 / 0 / 0 | 10.68 | 8.91 |
| pilot1 | fixed_two_stop | 25 / 0 / 0 | 22 / 3 / 0 | 16.19 | 15.65 |
| pilot1 | greedy_crossover | 14 / 0 / 11 | 12 / 2 / 11 | 23.77 | 11.87 |
| d2 | fixed_one_stop | 25 / 0 / 0 | 24 / 1 / 0 | 8.13 | 8.09 |
| d2 | actual_strategy | 23 / 0 / 2 | 23 / 0 / 2 | 7.95 | 7.59 |
| d2 | fixed_two_stop | 25 / 0 / 0 | 25 / 0 / 0 | 23.02 | 22.58 |
| d2 | greedy_crossover | 0 / 4 / 21 | 1 / 3 / 21 | | 47.70 |

d2's fixed_one_stop invalid row breaks the limit in its second (SOFT) stint, which moving
the stop earlier cannot fix. d2's four greedy_crossover invalid rows on main were
TEST_UNKNOWN plans; under R8 three are invalid for the limit. No pilot1 or d2 row has
negative regret.

## Milestone 2 numbers (v1, deterministic, before R8)

The Milestone 2 table is `reports/eval/v1_baselines_split.csv` (summary
`reports/eval/v1_baselines_split_summary.csv`; `python -m src.eval report
reports/eval/v1_baselines_split.csv` prints it): belief `to_snapshot`, world the default
SimConfig with the full window. `reports/eval/v1_baselines.csv` is kept as the record from
before the split (default SimConfig, full window for both planning and scoring). Both,
overall:

| policy | scored before | scored after | mean regret s before | after | median before | after | p90 before | after | stop agreement before | after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 250 | 4.95 | 4.95 | 3.01 | 3.01 | 13.60 | 13.60 | 0.90 | 0.90 |
| actual_strategy | 199 | 199 | 6.72 | 6.72 | 7.11 | 7.11 | 14.52 | 14.52 | 0.57 | 0.57 |
| fixed_two_stop | 211 | 211 | 12.95 | 12.95 | 9.25 | 9.25 | 21.18 | 21.18 | 0.00 | 0.00 |
| greedy_crossover | 123 | 185 | 18.77 | 11.91 | 18.61 | 11.54 | 23.85 | 20.14 | 0.06 | 0.70 |
| baseline agent (recorded) | 1 | 1 | 6.29 | 6.29 | | | | | 0.00 | 0.00 |

Only greedy_crossover moves: it is the one baseline that reads the fitted pace. The fixed
policies and the actual strategy depend only on the snapshot and the modelled compounds,
which are the same in both fits, and the recorded plan is fixed. Every reference plan is
the same. On the 123 scenarios greedy_crossover scored both times, 60 have lower regret
under the belief and none higher; on 62 more it no longer stops immediately. Under the
hindsight fit, the race-fitted linear MEDIUM wear (0.068 s/lap, the curvature #26
explained) brought its crossover early; on the to_snapshot fit it does not. A baseline
planned on hindsight doing worse than one planned on the belief is the direction an honest
eval should show.

Before the split, from `reports/eval/v1_baselines_summary.csv`:

| policy | scored | mean regret s | median | p90 | stop agreement |
| --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.95 | 3.01 | 13.60 | 0.90 |
| actual_strategy | 199 | 6.72 | 7.11 | 14.52 | 0.57 |
| fixed_two_stop | 211 | 12.95 | 9.25 | 21.18 | 0.00 |
| greedy_crossover | 123 | 18.77 | 18.61 | 23.85 | 0.06 |
| baseline agent (recorded, 1 scenario) | 1 | 6.29 | | | 0.00 |

The baseline agent row is the recorded run in `docs/sample_run.txt` (HAM from lap 0), scored
under this simulator rather than the legacy linear one it was graded on. The live agent
(`--policies agent`) now answers every scenario through the snapshot loop.

Things the numbers show that the owners may want to look at:
- `greedy_crossover` stops immediately (a zero-lap first stint, R1) on 141 of 264 scenarios
  before the split and on 79 under it, so it is unavailable there.
- `actual_strategy` is unavailable on 65 scenarios where a real stint was shorter than the
  10-lap minimum v1 carries. Sets sampled under R2's default of 3 keep those strategies.

## Position regret on v1 (world traffic on)

`reports/eval/v1_world_baselines.csv` (summary beside it): v1, the baselines, belief
default, world `configs/sim/world.yaml` (full window, ghost traffic, events none, so no
drawn-events warning). The world model changes, so the reference and the time regrets move
too:

| policy | scored | mean regret s | median | stop agreement | mean position regret | mean final position |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 5.74 | 3.68 | 0.88 | 0.34 | 6.88 |
| actual_strategy | 199 | 7.45 | 7.99 | 0.58 | 0.32 | 7.46 |
| greedy_crossover | 185 | 12.85 | 11.78 | 0.68 | 0.64 | 7.02 |
| fixed_two_stop | 211 | 13.93 | 10.65 | 0.02 | 0.82 | 7.15 |

The policy order by position regret is the order by time regret except that actual_strategy
and fixed_one_stop swap (0.32 against 0.34). Per row, the signs of `position_regret` and
`regret_s` agree on 332 of 845 scored rows (actual_strategy 86/199, fixed_one_stop 65/250,
fixed_two_stop 101/211, greedy_crossover 80/185). All 513 disagreements are one kind: time
lost (`regret_s` > 0) with the same finishing position (`position_regret` 0), median 7.8 s
of regret against 14.1 s on the rows that lose a place. There is no row with opposite
signs, no row that loses a place without losing time, and no negative position regret. The
world is deterministic (one draw), so positions are whole numbers: a few seconds rarely
cross a gap to the next car.

That shape is expected, not a defect. With one draw a position is an integer, and in a
spread-out field a few seconds rarely move a place, so most rows that lose time finish where
the reference does: 513 of 845 lose time at the same position, and the signs agree on 332.
Position regret becomes informative with `n_samples` above 1 (the mean over draws moves in
fractions of a place) and where the field is close, after an SC, which is exactly where the
SC-bunching limit applies (docs/traffic.md, limits 1 and 5). Read it beside
`final_position_mean`, not instead of the time regret.

## The learned calibrator

The belief's prior for its `to_snapshot` fit comes from a calibrator (`src/sim/calibrators.py`).
The default, `shrinkage`, is the practice fit plus config values with fixed taus; it learns
nothing across races. The ridge calibrator (SPEC 0024, #60) learns each factor from features
known at the calibration point k, across races. It is now trained on the real calibration
races, compared with shrinkage, adopted per factor and committed. It is not the default (below).

**What is trained, on what.** One ridge per factor on `build_dataset`'s rows: one row per
(race, k) with the k snapshot laps of the local profile (every 5 laps, plus the laps around
SC and VSC periods). The features are known at k (total laps, pit loss, mean residual and
laps so far per compound against the practice prior, SC and VSC laps so far, the track
temperature trend). The targets are the race's hindsight factors (`hindsight_targets`: the
full-window fit's base delta, wear and fuel, and the observed event factors). The training
races are the seven calibration races, given explicitly: 2021_04_spain, 2023_07_spain,
2023_01_bahrain, 2019_10_united_kingdom, 2022_02_saudi_arabia, 2024_07_emilia_romagna,
2025_23_qatar. That gives 85 rows. A held-out race (`data/eval_set.json`) is refused before
anything about it is loaded, and the training code never reads one; a test checks this.

**One command** runs the evaluation and the training, and writes the report:

    python -m src.sim.learned_calibrator train      # --jobs N, --resume reuses current folds

It runs these steps:

1. Leave-one-race-out folds (`cross_race.fold_job`) for none, shrinkage, ridge (degradation
   targets) and ridge_all (event factors too). Each arm is the belief simulator as
   `configs/sim/belief.yaml` stands (`configs/sim/arms/belief_*.yaml`, `base_config`):
   posterior draws, residual lap noise, the wear bound, and traffic off. G replays the race's
   real SC and VSC periods. A fold's ridge is fitted on the other six races only, inner
   alpha search included (the #60 leakage test still holds). Windows: `pre_race` and
   `to_snapshot`.
2. The per-factor table, then the adoption rule.
3. The hybrid (`configs/sim/arms/belief_hybrid.yaml`, written by the command), run through
   the same folds.
4. The belief measures and the default rule.
5. The trained artifacts and `configs/sim/belief_learned.yaml`.
6. The report: `reports/calibration/learned/summary.md` and its csvs. The fold outputs stay
   under `reports/calibration/learned/work/`, which is gitignored.

**The adoption rule** was stated in the code before any result. It is applied per factor
kind, pooled over compounds, because the engine takes one prior sd per kind. Ridge is
adopted for a kind when its leave-one-race-out rmse is below shrinkage's and its coverage is
no worse. Coverage is no worse when |coverage - nominal| is no larger at both 68 and 90
percent. Both conditions must hold in both belief windows. Degradation kinds are read from
ridge, event kinds from ridge_all. Every other kind stays on shrinkage.

| factor | window | rmse shrinkage | rmse ridge | cov68 shr / ridge | cov90 shr / ridge | adopted |
|---|---|---|---|---|---|---|
| base_delta_s | pre_race | 1.702 | 1.921 | 0.33 / 0.76 | 0.40 / 0.83 | no |
| base_delta_s | to_snapshot | 0.693 | 0.771 | 0.25 / 0.31 | 0.45 / 0.49 | |
| wear_s_per_lap | pre_race | 0.0693 | 0.0858 | 0.42 / 0.64 | 0.63 / 0.64 | no |
| wear_s_per_lap | to_snapshot | 0.0596 | 0.0691 | 0.41 / 0.54 | 0.61 / 0.67 | |
| fuel_s_per_lap | pre_race | 0.0304 | 0.0195 | 0.13 / 0.58 | 0.29 / 0.71 | **yes** |
| fuel_s_per_lap | to_snapshot | 0.0222 | 0.0211 | 0.24 / 0.35 | 0.41 / 0.55 | |
| pit_loss_s | pre_race | 0.723 | 0.792 | 0.00 / 0.87 | 0.00 / 1.00 | no |
| pit_loss_s | to_snapshot | 0.723 | 1.003 | 0.00 / 0.99 | 0.00 / 1.00 | |
| lap1_extra_s | pre_race | 1.831 | 2.875 | 0.00 / 0.68 | 0.00 / 0.68 | no |
| lap1_extra_s | to_snapshot | 1.831 | 2.826 | 0.00 / 0.56 | 0.00 / 0.68 | |
| sc_lap_factor, vsc_lap_factor, sc_pit_loss_factor | both | = | = (fallback) | 0 / 0 | 0 / 0 | no |

Only the fuel slope is adopted. Ridge's base and wear are less accurate than shrinkage's;
its wider sds only buy coverage. The SC and VSC factors have fewer than 5 training races
with a period, so ridge falls back to shrinkage in every fold. Shrinkage carries the event
factors at sd 0 (the config or race value), so its event coverage is 0 by construction.
`factors.csv` has none and the hybrid too, and `factors_by_compound.csv` splits by compound.

**The belief measures** are leave-one-race-out over the seven races and race-weighted, with
95% race-bootstrap CIs:

- G: the real remaining plan replayed from each calibration point, with the pace-corrected
  error at horizons 10-49.
- Stop-gain: the predicted gain against the real gain at the teams' real pit stops.
- G's interval coverage.

`to_snapshot` is the belief window.

| window | calibrator | G mae s [CI] | G bias s | G diff vs shrinkage [CI] | G cov68 / cov90 | stop-gain mae s | stop-gain bias s | stop-gain diff [CI] |
|---|---|---|---|---|---|---|---|---|
| to_snapshot | none | 26.12 [16.92, 35.13] | +15.39 | +0.71 [-4.25, 5.67] | 0.14 / 0.21 | 0.733 | -0.26 | -0.10 [-0.19, -0.01] |
| to_snapshot | shrinkage | 25.40 [18.58, 31.50] | +13.06 | | 0.18 / 0.27 | 0.837 | -0.37 | |
| to_snapshot | ridge | 31.47 [21.34, 41.30] | +12.17 | +6.07 [0.49, 12.28] | 0.22 / 0.34 | 1.009 | -0.34 | +0.17 [-0.01, 0.33] |
| to_snapshot | ridge_all | 31.16 [21.06, 41.00] | +11.38 | +5.75 [0.08, 12.21] | 0.23 / 0.36 | 1.009 | -0.34 | +0.17 [-0.01, 0.33] |
| to_snapshot | hybrid | 25.05 [18.29, 31.01] | +0.86 | -0.35 [-3.56, 2.93] | 0.22 / 0.34 | 0.842 | -0.13 | +0.005 [-0.04, 0.05] |
| pre_race | none | 26.12 [16.92, 35.13] | +15.39 | -1.48 [-3.19, 0.10] | 0.14 / 0.21 | 0.733 | -0.26 | 0.00 |
| pre_race | shrinkage | 27.59 [18.15, 36.83] | +18.72 | | 0.49 / 0.68 | 0.733 | -0.26 | |
| pre_race | ridge | 45.00 [34.41, 60.46] | +38.37 | +17.40 [1.94, 33.85] | 0.83 / 0.96 | 1.19 | +0.08 | +0.45 [0.05, 1.15] |
| pre_race | hybrid | 23.61 [19.16, 27.98] | +1.40 | -3.98 [-11.98, 4.67] | 0.65 / 0.89 | 0.70 | -0.14 | -0.03 [-0.08, 0.01] |

Per race numbers are in `belief_by_race.csv`, and G by 10-lap horizon bucket in
`g_by_horizon.csv`.

**The default rule** was also stated before any result. The hybrid is tried first, then plain
ridge. A candidate becomes the belief's default when, on the `to_snapshot` race-weighted point
estimates, all three hold:

- its G mae is below shrinkage's;
- its stop-gain mae is no worse;
- its G coverage is no worse (|coverage - nominal| at 68 and 90 percent).

With 7 races, a CI that excludes 0 is not required.

**Result: not adopted; shrinkage stays the default.** The hybrid changes only the fuel prior.
It replays the real strategies slightly more accurately: G mae 25.05 against 25.40 s
(paired -0.35 s, CI -3.56 to +2.93). It removes most of shrinkage's signed bias (+0.86
against +13.06 s), and it covers better (0.22 / 0.34 against 0.18 / 0.27). It fails one
condition, by 0.005 s: its stop-gain mae is 0.8417 s against 0.8368 s (paired CI -0.04 to
+0.05). Both differences are well inside noise.

Plain ridge is worse on G (+6.07 s, CI 0.49 to 12.28) and on stop-gain. So learning the base
and wear across seven races does not beat the practice-plus-race shrinkage fit.

The learned calibrator ships as an option, `--belief-config configs/sim/belief_learned.yaml`,
which the command writes. Every interval is too narrow under every calibrator: G's 90% coverage
is at most 0.36 in the belief window. That is a separate problem from the calibrator's prior.

**Trained artifacts, deployed without leakage.** The artifacts live in
`models/calibrators/ridge_hybrid/`:

- `loo/<race_id>.json`, one per calibration race, fitted on the other six races.
  It is identical to that race's fold model.
- `all.json`, fitted on all seven.
- `manifest.json`: the command, the race list, the commit, `sim_code_version`, the params,
  the per-factor sources, the adoption and the decision.

The files are the JSON `RidgeCalibrator.save` writes. The calibrator `trained`
(`calibrators.TrainedCalibrator`, params `artifacts` and `sha256`) loads by race id. A
calibration race gets its leave-one-out model, used when it is evaluated as development
data. Any other race gets `all.json`; that covers the held-out races, which never enter
training. A race id that is not canonical is refused, because `all.json` under another name
would leak. A model whose `train_races` holds the race it would calibrate is refused too.

Nothing trains at evaluation time. `sha256` pins the artifacts, so the config hash names them.
The agent's model report carries `prior_sources`, {factor: "ridge" or "shrinkage"}, for this
calibrator.

**Caveat: not strictly as-of.** A leave-one-out model learns from races both before and
after the race it calibrates; for example, 2019 Britain is calibrated by a model trained on
2021-2025. Nothing from the evaluated race itself is used, but this is not what a team
could have known on that weekend.

A strictly as-of variant would train race R's model on `race_ids.as_of(R, races)` only.
With these seven races, 2019 Britain has no earlier race and 2021 Spain has one. Every
factor needs `RIDGE_MIN_RACES` (5) training races, so both would fall back to shrinkage
entirely. `--as-of` is a stub that says so; it needs more seasons of calibration races.

**Pending: the tyre-life change.** A separate change, `fix/tyre-life-no-hindsight`, changes
the wear model: a cliff, driver effects and hinge, a fuel-identification fix and a warm-up
term. That changes the hindsight targets and what some factors mean. The numbers above are
on the simulator before that change. When it lands, rerun the one command, which regenerates
the arm, the artifacts, `belief_learned.yaml` and the report.

## Limitations

- **The world is fitted on hindsight.** It reads every race lap (`full` window); only the
  belief is held to the laps up to the snapshot (above, "Who sees what").
- **Tyre life is a cliff, priced, not a limit** ("The wear cliff"). The R8 stint limit that read
  the whole race is retired; results scored under it are superseded ("Result files"). The
  world's cliff age is the oldest age raced (hindsight, allowed for the scorer); the belief's is
  an as-of estimate whose prior is a share of race distance from earlier races, so early in a
  race real strategies can run past its point estimate (on the calibration races 13 percent of
  the teams' remaining plans have a stint past it, median 2.7 laps; measured with the
  belief's point edge before the history edge and the capped form were settled).
- **2025 Qatar's mandated 25-lap maximum stint is not modelled.** It is a published regulation
  and would be a legitimate rule; R8 happened to stand in for it. Without it the world's lap-0
  reference there is a one-stop against the field's two.
- **A proxied compound's price has no race laps behind it** ("Proxied compounds"): its pace
  and wear are a step from a raced compound measured in practice, qualifying or other races,
  with the validation errors listed there, and its world cliff age is a practice age. At 2021 Spain
  the ordering constraint makes the proxied HARD a copy of SOFT, which 66 of v1's references
  use as a second SOFT-like compound; rows carry `proxied_compounds` and
  `opt_proxied_compounds` so results can be read with and without the rows that rest on one.
- **v1 scores under a 10-lap minimum stint** (its snapshots predate R2's 3), which makes
  actual_strategy unavailable on 65 scenarios.
- **The SC and VSC rates are provisional** and Spain-derived (configs/sim/belief.yaml). The
  default world draws no events; the opt-in hybrid world (configs/sim/world_hybrid.yaml) draws later
  periods with the same rates ("Safety cars and VSC in the scoring world"). Position regret has the
  traffic limits of docs/traffic.md.
- **d3 is rebuilt at the final evaluation**; its committed reference and baselines predate R8
  ("Reference cache").

## Running on Sol

`python -m src.eval jobs` installs the eval runner and hands everything after `--` to
`python -m src.sim.jobs`, so `eval_chunk` tasks score slices of a scenario set:

```bash
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    manifest eval_chunk --profile sol --n-items 264 --manifest $OUT/manifest.json --out $OUT
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    eval_chunk --profile sol --manifest $OUT/manifest.json --task-id 0 --out $OUT
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines -- \
    aggregate eval_chunk --profile sol --manifest $OUT/manifest.json --out $OUT
```

`slurm/eval_chunk.sbatch` uses the `python -m src.eval jobs ... -- eval_chunk ...` form
(set `EVAL_SCENARIOS`, optionally `EVAL_POLICIES`, `EVAL_WORLD_CONFIG`,
`EVAL_BELIEF_CONFIG`). `run-all` works with worker processes too:
`install_job_runner` registers its spec (scenarios path, policy names, recorded files,
config paths) with `jobs.set_eval_runner`, `run-all` carries it in every `eval_chunk`
task's payload, and each worker installs the runner before its task, under any start
method:

```bash
python -m src.eval jobs --scenarios data/scenarios/v1.json --policies baselines \
    --world-config configs/sim/world.yaml -- run-all eval_chunk --manifest M --out O
```

The runner reads cached references under `--reference-dir` (default `data/references`)
when they match the seed (0) and the profile's `n_samples` and `max_stops`; a report run adds
`--require-reference` so a missing reference fails the task (see "Reference cache").

LLM policies stay on the desktop.

## Simulator fidelity

The framework. There are two simulators. The JUDGE (the `world` above) scores every plan and
builds the reference: it should be as close as possible to what really happened, while using as
little hindsight as it can. The PREDICTIVE simulator (the `belief`) is what the agent, its tools
and the rule baselines plan with at a snapshot: it must use no hindsight, and its goal is to be
as close as possible to the judge under that restriction. Hindsight facts are measurement
targets for the model's components, not inputs: each component predicts from allowed
information and the whole-race fact says how good the prediction was. Every modelling factor
(wear shape and cliff, driver effects, fuel, warm-up, pit loss, safety cars, traffic,
calibrators, compound proxies, tyre-life priors) is kept or dropped by what it does to these
distances, measured by one command that takes any configuration and changes no lap time:

```bash
python -m src.sim.fidelity run    [--world FILE|default] [--belief FILE] [--world-overlay Y ...] \
    [--belief-overlay Y ...] [--races ...] [--out reports/calibration/fidelity] [--label main]
python -m src.sim.fidelity ablate [--spec configs/sim/fidelity/ablation.yaml] [same options]
python -m src.sim.fidelity report [--out ...] [--label main]
python -m src.sim.fidelity ledger [--world ...] [--belief ...]
```

Defaults: the calibration races (`fidelity.CALIBRATION_RACES`: the first six, 2021 and 2023 Spain,
2023 Bahrain, 2019 Britain, 2022 Saudi Arabia, 2024 Imola, and thirteen eligible, non-held-out
2021-22 weekends; never `auto`), the eval's default world as the judge
(`score.default_world_config` of the race's config, with the race's whole-race pit loss,
`score.world_pit_loss`, as `score.race_simulators` builds it), `configs/sim/belief.yaml` as the
predictive simulator. The ablation's default factors (configs/sim/fidelity/ablation.yaml): traffic
in the judge (`configs/sim/world.yaml`), in the belief (`configs/sim/belief_traffic.yaml`) and
both; hybrid SC/VSC in the judge; the judge's wear fit (driver effects and hinge, the default;
driver effects only; hinge only; neither); the judge's cliff edge (history only, history
extended by this race's ages, the default, or this race); the cliff off; legacy R8; driver
effects and hinge in the belief; the wear bound; the proxy; the belief's pit loss; the trained
calibrator. `--field-replay` adds `src/sim/field_replay.py`'s whole-field replay (cars run
together) with its own summary. A judge with `recorded` or `hybrid` SC/VSC replays whole races as
`recorded` (every period with its recorded lap times; from lap 0 a hybrid world's periods are
all recorded ones) and scores the plan panel as configured. The B snapshots are the policy's
(`calibration.snapshot_as_of`: scheduled distance, nominated compounds), for drivers who ran the
scheduled distance. Options: `--drivers` (B drivers per
race, 6), `--every` (B snapshot spacing, 5 laps), `--samples` (draws, 20), `--coverage` (100
draws), `--n-one` / `--n-two` (panel density), `--no-lodo`, `--no-loso`,
`--no-grid-references`, `--no-hindsight-variants`, `--no-components`, `--no-future-blindness`,
`--jobs`, `--only` (ablate: run only the named factors beside the base, `all_on` to keep it), `--cache` (default `<out>/.cache`, keyed by race, section, config hash, code version
and options, so an ablation reuses the base and a belief-only change reuses the judge's
Section A). Every race in `data/eval_set.json` is refused before anything is loaded, with no
override. A race or configuration that fails is reported and left out, never fatal. Races run
in spawned worker processes, `--jobs` of them (default min(6, cpu_count // 2)), each with one
BLAS thread (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` = 1, set before numpy
loads in the worker): the races are the parallelism, so a shared machine is not oversubscribed.

Selection rule, stated before any result: finishing ORDER matters more than exact time. Primary
= close-pair order accuracy of the judge out of sample (`A_close_pair_acc`); then relative gap
error (`A_rel_gap_mae_s`); then plan-ranking agreement between predictive and judge
(`B2_rank_rho`, `B2_regret_s`); absolute time is a tie-breaker. A factor that improves order or
relative gaps while worsening absolute time counts as an improvement.

Measures (each one line; `pace` and `fit` columns say what every number was fitted on):

| measure | definition |
| --- | --- |
| close-pair accuracy | pairs of cars on the same lap count whose real finishing gap is under 10 s: share whose simulated order is right |
| Spearman, mae_pos, within_1 | simulated against real finishing order (laps driven first, then simulated total), rank correlation, mean absolute position error, share within one place |
| rel_gap_mae_s | per car on the lead lap, time error less the race's mean error, mean absolute value (relative gaps, common offset removed) |
| abs_mae_s, bias_s | simulated less real total time per car, mean absolute and mean |
| fit | in-sample (every car's laps), leave-one-driver-out (the judge refitted without the car predicted: `exclude_snapshot_driver`, its pit loss re-estimated from the other cars' green stops), leave-one-stint-out (the stint's rows marked unclean, so no fit uses them; only that stint is reported), as-of (the belief's own fit at k) |
| pace | insample (the car's whole-race median clean-lap residual: one number, reads its own laps), practice (this weekend's long runs), asof (green laps up to k, practice under five laps), none (the field-mean car) |
| O version 1 / 2 / 3 | the judge as configured; the judge with each whole-race ledger input replaced by its substitute, singly and all together; the predictive simulator replaying every car's real remaining strategy from k = 0, 10, 20, 30, 40 (real SC/VSC laps ahead, real time to k) |
| A1 | whole-race replay of every car with at least 90% of the winner's laps: race time error raw and pace-adjusted, per race and pooled |
| A3 stint | simulated less real time of every real stint, lap 1, in- and out-laps left out |
| A3 lap residuals | real less predicted lap time, pace removed, on clean laps, by compound and tyre-age bucket; `late_stint_resid_s` over the last 5 laps of each stint |
| A3 stop gain | predicted less real gain from fresh tyres at real stops (last two clean laps before, first two after) |
| A3 pit | error on the in- and out-laps including the charged loss, per stop |
| A3 neutral | error on SC and VSC laps |
| A4 | the judge's lap-0 reference (optimize_batch) per field car: its stop count against the field's mode and the car's own; the share of reference plans (lap 0 and every B snapshot) with a stint past the oldest age any car ran on that compound (`beyond_raced`), or past it plus the R8 margin |
| A6 ledger | every input of the judge and the predictive config, tagged this race whole / this race to snapshot / weekend / history / regulation / a priori / unclassified; derived from the config and this branch's code paths, extended or overridden by an `x_ledger:` list in a config file; a non-default config field without a rule shows as unclassified |
| A paired | per whole-race ledger input with a substitute (`fidelity.SUBSTITUTES`: pre_race fit for the full-window fit, the pre-race prior from earlier races (`pit_loss.pit_loss_prior`) for the whole-race pit loss, the rate model for recorded/hybrid SC/VSC, the historical prior alone for the wear-cliff edge, R8 off where it is on, the proxy off, traffic off), the change in order, gaps, time, stop gain, pit error and A4 against the judge as configured |
| B panel | per snapshot (every 5 laps, 6 sampled drivers): the real remaining strategy, stay out, one stop at 4 laps on every compound, two stops at 3 first-stop laps on every compound pair |
| B2 | over the plans legal under both configs' own rules: Spearman of predictive against judge times, the judge time the predictive favourite loses to the judge favourite (regret), agreement on pit-now against stay; `_green` the same with no SC/VSC on either side |
| B1 | predictive less judge remaining time per panel plan (under R1-R3), by horizon, phase and plan kind; `green` without SC/VSC |
| B3 | share of plans whose judge time lies in the predictive 68% / 90% interval, the same for gaps to the real plan; per fitted factor (base, wear, fuel) whether the judge's value is within 1 / 1.645 predictive sds |
| B4 | `calibration.midrace` (the G measure) under the belief at the version-3 laps |
| B5 | `tests/future_blindness.py` run on its synthetic race with this belief, when the branch has it; else "not checked" |
| D | per component (`fidelity.COMPONENTS`: tyre life (`tyre_history.tyre_life_prior` against the oldest age run, the median stint-end age and `observed_tyre_life`'s edge), wear rate, fresh-tyre gain, compound gap, fuel, pit loss (the prior from earlier races and `pit_loss.estimate` at k against the world's whole-race median, green and SC), lap 1 and warm-up, traffic rejoin, neutralisations (rates), neutralised laps (lap time under SC/VSC)), each predictor from allowed information against the whole-race target: error, bias, coverage where an sd is stated. A branch adds one `register(Component(...))`; code absent on the branch reports "not present" |
| C | the scorecard for the base, each factor overlay alone and `all_on` (configs/sim/fidelity/ablation.yaml); headline changes pooled with a race-cluster bootstrap 95% interval over the races every configuration ran, and the ledger change. A factor whose switch is not on the branch (a module or file it `requires`, or an overlay SimConfig rejects) is skipped with a note |

Before numbers on main (2026-11-22, f7be251: wear cliff, R8 retired, traffic off by default) are in `reports/calibration/fidelity/main/scorecard.md`
and the ablation in `reports/calibration/fidelity/ablation/ablation.md`, both computed on main f7be251
over 18 calibration races (2022_11_austria did not load then).

### What the ablation says (f7be251, selection rule applied)

Changes against the base (the default world and `belief.yaml`), pooled, race-cluster 95% intervals:

- **Adopt-candidates.**
  - Hybrid SC/VSC in the judge: close-pair order +0.016 [0.000, +0.038], relative gaps −2.4 s [−4.4, −0.5], regret −0.36 s, 90% coverage +0.46. It is the only factor that clearly improves order and gaps together.
  - The judge's wear fit without the hinge (driver effects only): close pairs +0.025 [−0.003, +0.060], rank agreement +0.010, regret −0.48 s.
  - Driver effects in the belief: rank agreement +0.031 [+0.007, +0.053], regret −1.0 s [−1.8, −0.3], predictive order +0.021.
  - The history-only tyre-life edge in the judge: close pairs +0.025 [0.000, +0.060], and no hindsight.
- **Keep.**
  - The wear cost past the edge: switching it off costs rank −0.13 and regret +5.4 s.
  - The belief's wear bound: removing it costs regret +6.9 s.
- **Reject.**
  - The learned (trained) calibrator: regret +0.76 s [+0.18, +1.57], though coverage improves.
  - The hinge in the judge: driver effects alone or neither do better on close pairs.
- **Undecided: traffic.** In the per-car replay the other cars are their recorded trajectories, so traffic cannot change order there; it costs regret +0.4 s. It is judged on the whole-field replay (`src/sim/field_replay.py`) instead.

## Tests

`tests/test_eval.py` runs offline: synthetic races for sampling and scoring, plus the
committed Spain cache for the v1 set and an end-to-end baseline run.
`tests/test_eval_integration.py` covers the belief and world split and the decision unit,
`tests/test_eval_part2.py` positions, the drawn-events warning, the disagreement flag, the
eval split, `run-all` with two spawned workers, the agent policy, the snapshot contract and
the minimum-stint rule. `tests/test_eval_reference.py` covers the reference cache (key
stability, cached against computed rows, `--check` across code versions, hindsight
events, the d3 exception), `regret_se_s` on a world with known per-draw differences,
`within_mc` at N = 1 against the tie rule, and the R8 stamps on rows and summaries.
`tests/test_optimizer_bridge.py` covers the limit-aware baselines (fixed_one_stop's moved
stop, greedy_crossover's forced stop, a plan still past the limit scored invalid, and every
baseline unchanged with R8 off). `tests/test_eval_rule_baselines.py` covers the rule
baselines on synthetic races and on 2021 Spain: the default `baselines` group unchanged, the
already-stopped and not-yet-stopped branches, a forced stop for R3 and for R8 (one stop and two),
the invalid row when no legal plan exists within `max_stops`, legality, determinism and regret
>= 0. `tests/test_sim_proxy.py` covers the compound proxy (which
sources enter the blend and with what weight, the weighted mean, anchoring, the qualifying gap
and its place in the model report, the ordering constraint and the wear floor, the proxied
covariance and draws, the belief's information boundary, R8 for an unraced compound, the off
switch, the committed table and weights, the leave-one-race-out weight helper) on synthetic
weekends, and on
2022 Saudi Arabia that the world prices SOFT and accepts the belief's continuations. Modules
that test other mechanics on synthetic two-compound races carry `pytest.mark.compound_proxy_off`
(tests/conftest.py): their configs default to the proxy off, so they pin those mechanics as
they were. `tests/test_sim_fidelity.py` covers the fidelity scorecard: a judge fitted on a
synthetic race drawn from its own model replays it exactly (in-sample, leave-one-driver-out and
leave-one-stint-out), a deliberately wrong belief shows a positive B1 bias and more stops in its
favourite, the right belief none, the held-out refusal, the ledger and its substitutes, the
component registry's "not present", the section cache, and (slow) a scorecard on 2021 Spain. No
test calls a model.

## Live agent runs for the Milestone 2 report

The live-agent outputs in `reports/eval/case_b/`, `reports/eval/pilot1*.csv` and
`reports/eval/trajectories/` were produced on ASU RC Voyager with `LLM_BACKEND=voyager`,
`VOYAGER_BASE_URL=https://openai.rc.asu.edu/v1`, `VOYAGER_MODEL=qwen3-30b-a3b` and the default
`VOYAGER_TEMPERATURE` (1e-9), all set through `.env` (the API key is never committed). Case B is
`python -m src.baselines.runner --driver HAM --lap 30 --json --quiet`, run ten times
(`reports/eval/case_b/run_01..10`); the pilot is `python -m src.eval score --scenarios
data/scenarios/pilot1.json --policies baselines agent`, and the paired intervals are
`python -m src.eval.compare reports/eval/pilot1.csv --reference agent --others fixed_one_stop
actual_strategy`. The model is not deterministic at this temperature: identical inputs gave 5 of
10 submissions on Case B. The pilot set is a development set, not a report-set estimate.

**Reruns after the submission-budget fix (#59).** `reports/eval/pilot1_fix*.csv`,
`reports/eval/d2_fix*.csv`, `reports/eval/case_b/fix_run_01..10.json` and
`reports/eval/trajectories/{pilot1_fix,d2_fix}/` and `trajectories/case_b/fix_run_*.jsonl` are the
same runs with the same settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2), repeated after #59 made a submission always accepted and
outside the 24-call evidence budget. The `*_fix.csv` files hold only the agent rows; the
`*_fix_combined.csv` files add the unchanged baseline rows of the Milestone 2 files (the
reference columns match row for row) and are the input to `*_fix_compare_regret_s_*`. The
Milestone 2 files above stay the cited ones; the reruns are a follow-up measurement.

**Reruns after the turn-limit grace (#72).** `reports/eval/pilot1_fix2*.csv`,
`reports/eval/d2_fix2*.csv` and `reports/eval/trajectories/{pilot1_fix2,d2_fix2}/` repeat the
pilot1 and d2 runs once with the same settings. The two sets ran one after the other, so the
latency figures are not affected by contention. Since #72 the agent loop prints a turn-status
line, refuses an evidence call identical to one already run, and follows the 10 evidence turns
with up to 2 grace turns that offer only `submit_recommendation`. The rows carry
`submitted_in_grace` and `evidence_calls`. Loop settings of these runs: status line on,
repeats refused (a mode that no longer exists; their `evidence_calls` excludes the refused
repeats and they have no `repeat_calls`), grace on. The current defaults (status line off,
repeats run again) are the loop of the `*_fix` runs plus the grace. The `*_fix2_combined.csv` files are built the same way
as the `_fix` ones: the new agent rows plus the Milestone 2 baseline rows, whose reference
columns match row for row. They are the input to `*_fix2_compare_regret_s_*`. The Milestone 2
files stay the cited ones.

**dev6, agent pass 1.** `reports/eval/dev6_agent_pass1.csv` is one pass of the agent over the 150
dev6 scenarios with the same settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2, main's world defaults: R8 at rank 1 and margin 3, driver effects
off), run as one sequential job (`score --scenarios data/scenarios/dev6.json --policies agent
--seed 0 --require-reference`) at commit 3223131, 94 minutes wall time, with no other agent
run at the same time. `score` logs trajectories under the set version; they were moved to
`trajectories/dev6_pass1/`. dev6 is a development set on calibration races, not a held-out
result. Loop settings of this pass: status line on, repeats served from the cache (today
`AGENT_TURN_STATUS=1 AGENT_REPEAT_MODE=cache`), grace on; its `repeat_calls` are cached
repeats, which its `evidence_calls` exclude.

**dev6, run 2.** `reports/eval/dev6_run2_agent.csv` is a second single pass over the same 150
scenarios with the same model settings (qwen3-30b-a3b on Voyager, default temperature, seed 0,
`n_samples` 50, `max_stops` 2, R8 at rank 1 and margin 3, driver effects off), one sequential
job (`score --scenarios data/scenarios/dev6.json --policies agent --seed 0 --n-samples 50
--max-stops 2 --require-reference`) at commit 55a67ee, 54.6 minutes wall time, with no other
agent run at the same time; trajectories were moved from `trajectories/dev6/` to
`trajectories/dev6_run2/`. `dev6_run2_baselines.csv` is `--policies baselines
belief_optimizer` on the same references. Three things differ from pass 1: (1) the world prices
unraced compounds by the weighted blend ("Proxied compounds"; reference
`data/references/dev6/fbb33ef732dd.parquet`, sim_code_version `58b50a00e0e3`, world config
`ccb08edf391b`), so 2022 Saudi Arabia's SOFT continuations are scored rather than rejected;
(2) the agent loop runs at main's defaults (status line off, repeats run again and counted
against the budget, grace on; `AGENT_TURN_STATUS` and `AGENT_REPEAT_MODE` unset), where pass 1
had the status line on and cached repeats, so run 2's `evidence_calls` include repeats and
pass 1's do not; (3) the `belief_optimizer` baseline is added. Pooled, the agent scored 150 of
150 (pass 1: 129, with 21 Saudi rows invalid for the unknown compound), mean regret 3.95 s
(median 1.68), against belief_optimizer 4.11, actual_strategy 7.15 and fixed_one_stop 9.00;
agent minus belief_optimizer is -0.16 s, 95% interval [-0.58, 0.20] with driver clusters (88
ties of 150). 119 submissions were voluntary and 31 came in a grace turn (pass 1: 74 and 76),
at 72k mean input tokens (pass 1: 100k). The PIT / STAY_OUT decision is the same as pass 1's on
142 of 150 scenarios. No row of any policy is invalid for anything but its own plan's R8 stint
limit. dev6 is a development set on calibration races, not a held-out result.

**dev6, run 3 (variance repeat).** `reports/eval/dev6_run3_agent.csv` is an identical repeat of
run 2, made only to measure run-to-run variation; **run 2 stays the cited development result**.
Same command, scenarios, reference (`fbb33ef732dd`), sim_code_version `58b50a00e0e3`, world and
belief config (`ccb08edf391b` / `0002e90cb6fe`), model (qwen3-30b-a3b on Voyager, default
temperature, seed 0) and agent loop (main's defaults, the four `AGENT_*` / `VOYAGER_TEMPERATURE`
overrides checked unset), at commit c685349 (no change under `src/` or `configs/` since run 2's
55a67ee or tag `d4-final-eval`), one sequential pass, 60.9 minutes wall time, nothing else calling
the model. Trajectories were moved from `trajectories/dev6/` to `trajectories/dev6_run3/`;
`dev6_run3_combined.csv` is the new agent rows plus run 2's baseline and belief_optimizer rows
(30 reference and stamp columns match row for row). Every episode submitted (110 voluntary, 40 in
a grace turn; run 2: 119 and 31) and there were no API errors; one row is invalid
(2021_04_spain:OCO:00, a lap-0 PIT whose plan has a 1-lap first stint, R2), so 149 of 150 scored.
Between the two passes the PIT / STAY_OUT decision agrees on 145 of 150 scenarios and the final
plan on 119; on the 149 scenarios scored in both, run 3 minus run 2 regret is +0.41 s per
scenario (sd 2.62, 95% interval [0.05, 0.90] with driver clusters), 121 of them identical. Pooled
mean regret moves 3.95 to 4.36 s, and the paired agent-minus-X differences move by +0.27
(fixed_one_stop: -5.28 to -5.01), +0.46 (actual_strategy: -3.25 to -2.79) and +0.40 s
(belief_optimizer: -0.16 to +0.24, interval [-0.21, 0.79]). So a headline number moves by about
0.3 to 0.5 s between identical passes: small against the agent's lead over fixed_one_stop and
actual_strategy, but as large as the agent-versus-belief_optimizer difference, which is a tie in
both passes and changes sign. Per race the movement is concentrated in 2023 Bahrain (+1.66 s),
2023 Spain (+0.60) and 2021 Spain (+0.39); four episodes there (the invalid one and three
STAY_OUT to PIT flips worth +4.4, +15.1 and +21.3 s) made three tool calls in the first turn, no
logged call in the next nine, and then pitted onto MEDIUM in a grace turn without evaluating a
plan. Without those four the per-scenario difference is +0.13 s [-0.06, 0.42].

**dev6, run 4 (turns with no usable tool call).** Why run 3's four episodes stalled, found by
logging every turn with no tool call and rerunning nine dev6 scenarios (the four, and five more
with 7 or 8 such turns in run 3) three times each with the run 3 loop: all 94 such turns were
replies that ended normally (finish reason `stop`, at most 571 of the 1024 output tokens) and
were JSON the parser rejected. 87 had one closing brace too many after `compare_plans`'
candidates (`"laps": 3}]}}}]}`), 4 one too few, 3 two whole envelopes in one reply; none was
cut off, empty, prose or fenced. The loop answered each with "No tool call was detected" and
did not show the model its reply, so it sent the same malformed JSON back: 72 of the 94 came
directly after another, and 5 of the 27 episodes stalled exactly as in run 3 (grace-turn PIT,
no plan evaluated). The fix (agent loop and Voyager backend, under "The agent policy" above):
those turns are logged, the reminder carries the parse error and where it failed, and a reply
whose only fault is its final run of closing brackets is read as the one run that closes
what is open (91 of the 94 had that shape; the 3 double envelopes are left to the reminder). The same 27 targeted episodes:
5 stalls before, 1 with the reminder alone, 0 with the whole fix; turns with no tool call 94,
50, 3; every decision STAY_OUT, as in run 2. Then one full pass, same command and settings as
runs 2 and 3, at commit e414a70, 66.5 minutes wall time, nothing else calling the model;
trajectories moved from `trajectories/dev6/` to `trajectories/dev6_run4/`.
`dev6_run4_combined.csv` is the new agent rows plus every non-agent row of
`dev6_run2_combined_rules.csv` (31 reference and stamp columns match row for row, floats to
1e-9 s: two `opt_time_s` differ by 9e-13).

| | run 2 | run 3 | run 4 |
| --- | --- | --- | --- |
| scored / invalid | 150 / 0 | 149 / 1 | 150 / 0 |
| mean / median / p90 regret s | 3.95 / 1.68 / 11.82 | 4.36 / 2.37 / 13.06 | 4.00 / 1.68 / 11.90 |
| voluntary / grace submissions | 119 / 31 | 110 / 40 | 121 / 29 |
| grace submissions with no plan evaluated | 0 | 4 | 0 |
| turns with no tool call (episodes with any; with 8 or more) | 148 (79; 0) | 222 (93; 6) | 23 (22; 0) |
| stalled episodes | 0 | 4 | 0 |
| Brier / pit-now agreement | 0.090 / 0.893 | 0.108 / 0.873 | 0.093 / 0.887 |
| evidence / repeat / tool calls per episode | 15.16 / 5.85 / 16.67 | 14.33 / 4.76 / 15.73 | 17.70 / 6.93 / 19.59 |
| latency s / input tokens / output tokens | 21.7 / 72.3k / 951 | 24.1 / 71.0k / 1098 | 26.3 / 84.1k / 966 |

Turns with no tool call are counted the same way for every run (a step with no call event);
run 4's 23 match its `no_tool_call` events: 13 a bare submission object without the
`tool_calls` envelope, 5 cut off at the 1024-token cap (many calls in one reply), 5 other
malformed JSON. 88 turns in 82 episodes were read after a closer repair. Agent minus X,
paired, 95% interval with (race, driver) clusters (run 3's rule rows paired the same way, not
committed):

| X | run 2 | run 3 | run 4 |
| --- | --- | --- | --- |
| actual_strategy | -3.25 [-5.50, -1.25] | -2.79 [-5.06, -0.68] | -3.20 [-5.47, -1.20] |
| belief_optimizer | -0.16 [-0.58, 0.20] | +0.24 [-0.21, 0.79] | -0.12 [-0.54, 0.26] |
| run_to_flag | -0.31 [-2.10, 1.35] | +0.07 [-1.80, 1.81] | -0.27 [-2.05, 1.37] |
| one_stop_total | -0.24 [-2.24, 1.56] | +0.14 [-1.96, 2.02] | -0.18 [-2.18, 1.59] |
| fixed_one_stop | -5.28 [-7.25, -3.30] | -5.01 [-7.03, -2.99] | -5.21 [-7.19, -3.24] |

So the stalls are gone and run 4 lands on run 2 (decision the same on 149 of 150 scenarios,
run 4 minus run 2 regret +0.04 s per scenario, 124 identical); the four run 3 episodes stay out
again (regret 0.00, 0.60, 0.00 and 5.38 s, against invalid, 4.96, 15.12 and 25.73 in run 3).
What got worse is cost: with calls that used to be lost now running, an episode makes 2.5 more
evidence calls and reads 16% more input tokens than in run 2, 4.6 s slower. The held-out d4
result (tag `d4-final-eval`) was produced by the loop before this fix; rerunning it is the
owner's decision.

### d4, the held-out result

d4 is the report set and this is the held-out result, declared in advance and run once at tag
`d4-final-eval` (commit cd1b937, main after #82), with no code or config change. Steps, each run
once: `scenarios --races 2019_06_monaco 2020_16_sakhir 2025_11_austria 2024_11_austria --version
d4 --max-per-race 25 --seed 0` (report set, `development: false`, 100 scenarios, 25 per race,
none skipped, digest `ae878eb91c28`, eval_set digest `98edff4380e1`); `reference` (seed 0, 50
samples, two stops: `data/references/d4/39bf1bbb4aa9.parquet`, sim_code_version
`58b50a00e0e3`, world config `ccb08edf391b`; `reference --check` clean apart from d3);
`score --policies baselines belief_optimizer --require-reference` (`d4_baselines.csv`); then,
after the baseline rows passed the gate (no row invalid for anything but an R8 breach of its own
plan, none for an unknown compound, no negative regret, all four races loaded and fitted), one
sequential agent pass (`score --policies agent --seed 0 --n-samples 50 --max-stops 2
--require-reference`, `d4_agent.csv`): qwen3-30b-a3b on Voyager, default temperature, seed 0,
main's agent loop (status line off, repeats run again, 2 grace turns, 10 evidence turns, 24-call
budget; `AGENT_TURN_STATUS`, `AGENT_REPEAT_MODE`, `AGENT_SUBMIT_GRACE_STEPS` and
`VOYAGER_TEMPERATURE` unset), R8 at rank 1 and margin 3, compound proxy on, 47.2 minutes wall
time, nothing else calling the model. It is one pass: the model is not deterministic at this
temperature, and no episode was rerun. d4's 25 Monaco scenario ids are d3's (same seed and per-race
count), so **d3's 2019 Monaco baseline rows of November 11 (`d3_baselines.csv`, pre-R8 simulator)
are superseded by d4's**; d3 itself is not rescored. What each race had been exposed to before
this run is the disclosure under "The registered split"; this run adds nothing to it beyond the
run itself.

Pooled over the 100 scenarios, the agent scored 100 of 100 (71 voluntary submissions, 29 in a
grace turn), mean regret 8.93 s (median 6.53), against belief_optimizer 8.95, actual_strategy
8.87 and fixed_one_stop 12.87. Paired, agent minus belief_optimizer is -0.03 s, 95% interval
[-0.29, 0.22] with driver clusters (39 ties of 100); agent minus actual_strategy -0.19 [-1.88,
1.43]; agent minus fixed_one_stop -4.28 [-6.73, -1.90]. Per race the agent's mean regret is 8.58
(2019 Monaco), 8.41 (2020 Sakhir), 2.55 (2025 Austria) and 16.17 s (2024 Austria). No row of any
policy is invalid for anything but its own plan's R8 stint limit, and the world prices every
compound from race laps at all four races (`proxied_compounds` and `opt_proxied_compounds` are
empty on every row), so the with-and-without-proxy reading is the same table. The belief does
price compounds without race laps at some snapshots (read from the model reports in the
trajectories): at 2024 Austria the belief prices SOFT from practice and the cross-race step at
24 of 25 snapshots (the 25th is lap 0, practice only), up to lap 67, while 23 of the 25 world
references, fitted on every race lap, use SOFT; every policy's regret there is about 16 s. Against dev6 run 2 (development, calibration races) the regrets roughly double for the agent
(3.95 to 8.93 s) and belief_optimizer (4.11 to 8.95), and the agent's lead over actual_strategy
(-3.25 [-5.50, -1.25] on dev6) is not there on d4.

### Rule baselines added after the held-out run

**Why.** `fixed_one_stop` always makes exactly one more stop from the snapshot, also after the
driver has made their stop, when the best plan is usually to run to the flag. Split by whether
the driver had already stopped, the agent was 7.4 s worse than fixed_one_stop on d4 before the
driver's first stop and 8.5 s better after it, so the headline "the agent beats a fixed
one-stop rule by 4.3 s" came from scenarios where the baseline makes a pointless stop. Two
sensible rule baselines were added beside it, `run_to_flag` and `one_stop_total` (rules in
"Rule baselines: run_to_flag and one_stop_total" above). They were added after the d4 results
were known.

**Order.** The rules were written in this file first, then implemented and tested, then scored
on the development set dev6 (below). No rule was changed after it was scored. They were then
frozen (code, docs, tests and dev6 outputs committed; the commit is given in the d4 part below)
and scored once on d4. No existing d4 or dev6 row was rescored.

**dev6 (development, calibration races).** `score --scenarios data/scenarios/dev6.json --policies
run_to_flag one_stop_total --seed 0 --n-samples 50 --max-stops 2 --require-reference`
(`reports/eval/dev6_rule_baselines.csv`, reference `fbb33ef732dd`, sim_code_version
`58b50a00e0e3`), compared with the committed dev6 run 2 rows
(`dev6_run2_combined.csv`) by `scripts/rule_baseline_tables.py`, which reads csv and scenario
json only and writes `dev6_run2_combined_rules.csv` (the run 2 rows plus the new ones; the
reference columns are checked equal first) and `dev6_run2_rules_compare_*`. Pooled:

| policy | scored / invalid / unavailable | mean regret s | median | p90 |
| --- | --- | --- | --- | --- |
| agent | 150 / 0 / 0 | 3.95 | 1.68 | 11.82 |
| belief_optimizer | 150 / 0 / 0 | 4.11 | 1.20 | 13.26 |
| fixed_one_stop | 123 / 27 / 0 | 9.00 | 7.28 | 20.57 |
| actual_strategy | 147 / 0 / 3 | 7.15 | 3.10 | 20.98 |
| run_to_flag | 150 / 0 / 0 | 4.27 | 0.05 | 10.93 |
| one_stop_total | 135 / 15 / 0 | 4.10 | 0.05 | 10.93 |

Agent minus each policy, paired on the rows both scored, 95% interval with (race, driver)
clusters (`eval.compare`, 2000 draws), wins / ties / losses from the agent's side:

| X | n | agent - X | 95% CI | W / T / L |
| --- | --- | --- | --- | --- |
| belief_optimizer | 150 | -0.16 | [-0.58, 0.20] | 26 / 88 / 36 |
| fixed_one_stop | 123 | -5.28 | [-7.25, -3.30] | 88 / 2 / 33 |
| actual_strategy | 147 | -3.25 | [-5.50, -1.25] | 70 / 37 / 40 |
| run_to_flag | 150 | -0.31 | [-2.10, 1.35] | 49 / 53 / 48 |
| one_stop_total | 135 | -0.24 | [-2.24, 1.56] | 46 / 47 / 42 |

Split by whether the driver has already stopped (`snapshot.stints` not empty), agent minus X:

| subset | X | n | agent - X | 95% CI |
| --- | --- | --- | --- | --- |
| not yet stopped (60) | fixed_one_stop | 45 | +3.96 | [1.86, 6.04] |
| not yet stopped | run_to_flag | 60 | +2.73 | [0.62, 4.59] |
| not yet stopped | one_stop_total | 45 | +3.96 | [1.86, 6.04] |
| already stopped (90) | fixed_one_stop | 78 | -10.61 | [-12.44, -8.75] |
| already stopped | run_to_flag | 90 | -2.34 | [-4.75, -0.17] |
| already stopped | one_stop_total | 90 | -2.34 | [-4.75, -0.17] |

On dev6 the agent ties both rule baselines pooled (intervals across 0); it loses to them before
the driver's first stop and beats them after it. `one_stop_total`'s 15 invalid rows are
`fixed_one_stop`'s own, on not-yet-stopped drivers (its plan there is fixed_one_stop's, row for
row: status, plan and regret), all R8 breaches of the second stint (12 at 2021 Spain, on the
proxied HARD and its practice limit of 21, 2 at 2022 Saudi Arabia, 1 at 2023 Bahrain), which
moving the stop earlier cannot fix. `run_to_flag` has no invalid row; it makes a stop on all 60
not-yet-stopped rows (the two-compound rule) and on 23 of the 90 already-stopped rows (20 where
the current set would pass its R8 limit before the flag, 3 where the driver had stopped onto the
same compound, so the two-compound rule still needed a stop), and runs to the flag on the other
67. No row of either has negative regret. Nothing was changed after this run:
it found no bug. Per race (`dev6_run2_rules_compare_per_race_regret_s_paired.csv`), the agent is
worse than both at 2022 Saudi Arabia (+3.3 and +3.7 s) and better than `one_stop_total` at 2021
Spain on its 13 scored rows; dev6 is a development set on calibration races, not a held-out
result.

**Freeze.** Commit `5a2bc41` (branch `feat/fair-rule-baselines`) holds the rules, the code, the
tests and the dev6 outputs above. No file under `src/`, `configs/` or `tests/` changed after it.

**d4 (held out), scored once, after the d4 results were known.** One command at the freeze
commit, the only command that loaded the four held-out races: `score --scenarios
data/scenarios/d4.json --policies run_to_flag one_stop_total --seed 0 --n-samples 50 --max-stops 2
--require-reference --out reports/eval/d4_rule_baselines.csv` (reference
`data/references/d4/39bf1bbb4aa9.parquet`, sim_code_version `58b50a00e0e3`). It ran without error
and was not rerun. No existing d4 row was rescored: the comparison rows are the committed
`d4_combined.csv`, and `scripts/rule_baseline_tables.py` (csv and scenario json only, the
reference columns checked equal) wrote `d4_combined_rules.csv` and `d4_rules_compare_*`.
`d4_combined.csv` is unchanged. Pooled:

| policy | scored / invalid / unavailable | mean regret s | median | p90 |
| --- | --- | --- | --- | --- |
| agent | 100 / 0 / 0 | 8.93 | 6.53 | 22.13 |
| belief_optimizer | 100 / 0 / 0 | 8.95 | 6.36 | 22.67 |
| fixed_one_stop | 94 / 6 / 0 | 12.87 | 16.76 | 26.28 |
| actual_strategy | 98 / 0 / 2 | 8.87 | 2.56 | 28.03 |
| run_to_flag | 100 / 0 / 0 | 9.26 | 6.63 | 24.59 |
| one_stop_total | 95 / 5 / 0 | 7.02 | 2.94 | 20.57 |

Agent minus each policy, paired, 95% interval with (race, driver) clusters, wins / ties / losses
from the agent's side:

| X | n | agent - X | 95% CI | W / T / L |
| --- | --- | --- | --- | --- |
| belief_optimizer | 100 | -0.03 | [-0.29, 0.22] | 29 / 39 / 32 |
| fixed_one_stop | 94 | -4.28 | [-6.73, -1.90] | 59 / 0 / 35 |
| actual_strategy | 98 | -0.19 | [-1.88, 1.43] | 30 / 28 / 40 |
| run_to_flag | 100 | -0.33 | [-2.14, 1.47] | 41 / 26 / 33 |
| one_stop_total | 95 | +1.66 | [-0.33, 3.65] | 35 / 26 / 34 |

Split by whether the driver has already stopped, agent minus X:

| subset | X | n | agent - X | 95% CI |
| --- | --- | --- | --- | --- |
| not yet stopped (30) | fixed_one_stop | 25 | +7.44 | [3.59, 11.42] |
| not yet stopped | run_to_flag | 30 | -0.18 | [-3.93, 2.96] |
| not yet stopped | one_stop_total | 25 | +7.44 | [3.59, 11.42] |
| already stopped (70) | fixed_one_stop | 69 | -8.53 | [-10.66, -6.37] |
| already stopped | run_to_flag | 70 | -0.40 | [-2.55, 1.81] |
| already stopped | one_stop_total | 70 | -0.40 | [-2.55, 1.81] |

Per race, agent minus run_to_flag / one_stop_total (`d4_rules_compare_per_race_regret_s_paired.csv`):
2019 Monaco +3.55 / +6.88, 2020 Sakhir +4.30 / +5.49, 2024 Austria -2.82 / -1.21, 2025 Austria
-6.37 / -5.42. The rule baselines are much better than the agent at Monaco and Sakhir and much
worse at 2025 Austria; a t interval over the four race means is [-8.51, 7.84] for run_to_flag
and [-7.76, 10.63] for one_stop_total.

What this says, plainly. On the held-out races **the agent does not beat a sensible rule**: it
ties `run_to_flag` (-0.33 s, interval across 0) and is 1.66 s worse than `one_stop_total` on the
95 scenarios that baseline scored, an interval that just reaches 0, so the agent is not shown
to be worse either. The agent's 4.3 s lead over `fixed_one_stop` is the pointless extra stop
that baseline makes after the driver's first stop: against a rule that does not make it, the
lead is gone, after the first stop (-0.40 s) as before it. Before the first stop, the agent
loses to the one-stop rule by 7.4 s, as it lost to `fixed_one_stop`, since the two are the same
plan there. `one_stop_total` has 5 invalid rows, `fixed_one_stop`'s own on not-yet-stopped
drivers (its plan there is fixed_one_stop's, row for row): the first stint passes its R8 limit
(2025 Austria SOFT, limit 19; 2024 Austria MEDIUM, limit 32) and no earlier stop keeps the one
remaining stint inside the second compound's limit. Those 5 rows are left out of its paired
comparison; the agent's regret on them is 1.9, 4.9, 24.8, 20.5 and 15.8 s. `run_to_flag` has
no invalid row and stops in 57 of 100 scenarios (25 one-stops and 5 two-stops before the first
stop, 26 one-stops and 1 two-stop after it). No row of either has negative regret. These are
one scoring of deterministic rules on four races; with the large race-to-race spread, the
pooled intervals are the ones to quote.

To print a Case B episode as a terminal transcript (for screenshots), replay a committed run with
`python scripts/show_case.py --replay reports/eval/case_b/run_01.json --commit 29d164c` (the commit
the run was produced at), or run one fresh episode with `--live`, which writes `case_b/live_<time>.*`.
