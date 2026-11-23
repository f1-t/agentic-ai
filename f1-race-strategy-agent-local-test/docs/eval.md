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
| belief | `configs/sim/belief.yaml` | `to_snapshot`: race laps up to the snapshot, shrunk toward the practice prior; traffic off; SC and VSC periods from the rate model with the rates fitted on the races before this one and the as-of history of period lengths (`events: rate`, `as_of`, `duration: history`); samples carry posterior parameter draws and residual lap noise, drawn from seed + `agent_tools.BELIEF_SEED_OFFSET` so the belief never plans on the world's samples (docs/rules.md R6) | every policy: the baselines plan on it through the bridge (`from_contract(snapshot, belief)`, `pace_for_crossover(belief, snapshot)`), and agent decisions are expanded on it |
| world | the profile's SimConfig with the eval world's wear fit and tyre-life edge (`config.world_default`) and SC/VSC on (`score.default_world_config`: events `hybrid`, the period in progress as recorded and later periods drawn from rates and lengths pooled over the dry races; "Safety cars and VSC in the scoring world"). This is `--world-config default`; `configs/sim/legacy/world_pre_scorecard.yaml` is the world before the fidelity scorecard (no SC/VSC) | `full`: every race lap (hindsight); traffic off; the whole-race pit loss ("Pit loss and track status") | the reference `optimize_batch` and every scoring call |

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
(docs/traffic.md has the rule). It is a switch on each side, **off by default**. The fidelity
scorecard (`python -m src.sim.fidelity`) measured it by the finishing-order measures, off and on,
and kept it off ("The current configuration (adopted from the scorecard)"):

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
should be measured again. Neither side is on by default: the fidelity scorecard measured it by
the finishing-order measures and kept traffic off ("The current configuration").

### Overtaking difficulty

The traffic model above uses one passing rule for every circuit. `src/sim/overtaking.py` counts
passes per circuit from the timing data and gives the ghost traffic a circuit-specific
threshold. It is a switch, `traffic.params.difficulty`, **off by default** (`none`), with the
earlier behaviour bit for bit when off.

**The count** (`observe_race`, `python -m src.sim.overtaking build`, written to
`data/derived/overtaking.csv`, `overtaking_pairs.csv` and the manifest; 28 cached races, never a
held-out one; 2021 Belgium is left out with 1 racing lap):

- A *close lap* is a car that starts lap L within 1.0 s of the car directly ahead on the same lap
  count (`data.gap_columns`). Both cars need a lap-L time, neither may be on an in-lap or out-lap,
  and lap L and the car ahead's lap L-1 must be green.
- A *pass* is a close lap on which the follower crosses the line first.
- Pit stops, lap 1, SC/VSC laps and the lap after them, retirements and lapping never count.
- The *rate* is passes per close lap. Normalising by how often cars are in range keeps a
  processional race at a wide track from reading as hard to pass. `passes_per_lap` (every order
  change on the same lap count) is reported beside it.
- A free-air pace fit per race (driver-stint intercept, compound age slope, a common lap slope,
  laps at least 2 s behind the car ahead) gives the follower's *pace advantage* on each close
  lap. A logistic P(pass) = sigmoid(slope (advantage - theta)) over the pooled close laps has
  slope 1.14 per s. The pass probability is 3% with no advantage, about 25% at 0.5-1.0 s,
  49% at 1.0-1.5 s and 78% at 2-3 s.
- `theta` is the advantage at which half the close laps end in a pass.
- The per-circuit value is the rate, in log-odds, shrunk toward the pooled rate with weight
  n / (n + 150 close laps). It becomes a threshold through the pooled logistic:
  theta = theta_pool + (z_pool - z) / slope.
- *Dirty air* is the pace-fit residual of laps under 1.5 s behind less that of free-air laps.
- *Sectors*: the boundary (S1, S2, line) at which the follower first runs ahead, from the
  cumulative sector session times.

| race | circuit | close laps | passes | rate | passes/lap | S1/S2/S3 | dirty air s | theta (own) s | judge rate / theta | as-of rate / theta |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019_10_united_kingdom | silverstone | 183 | 13 | 0.071 | 0.47 | 1/5/7 | 0.52 | 2.23 | 0.095 / 2.14 | (no earlier race) |
| 2021_01_bahrain | sakhir | 84 | 32 | 0.381 | 0.83 | 32/0/0 | 0.28 | 0.80 | 0.209 / 1.33 | 0.071 / 1.60 |
| 2021_02_emilia_romagna | imola | 103 | 19 | 0.184 | 0.56 | 16/1/2 | 0.52 | 2.07 | 0.119 / 1.92 | 0.169 / 1.60 |
| 2021_04_spain | catalunya | 188 | 34 | 0.181 | 0.67 | 30/2/2 | 0.66 | 1.71 | 0.222 / 1.27 | 0.173 / 1.60 |
| 2021_09_austrian | red_bull_ring | 254 | 25 | 0.098 | 0.37 | 2/23/0 | 0.43 | 1.85 | 0.131 / 1.82 | 0.176 / 1.40 |
| 2021_20_qatar | lusail | 158 | 35 | 0.222 | 0.76 | 34/0/1 | 0.50 | 1.80 | 0.101 / 2.08 | 0.151 / 1.33 |
| 2021_21_saudi_arabia | jeddah | 112 | 20 | 0.179 | 1.14 | 15/0/5 | 0.76 | 1.14 | 0.158 / 1.63 | 0.163 / 1.45 |
| 2021_22_united_arab_emirates | yas_marina | 98 | 10 | 0.102 | 0.28 | 0/9/1 | 0.54 | 1.95 | 0.170 / 1.55 | 0.165 / 1.45 |
| 2022_01_bahrain | sakhir | 151 | 40 | 0.265 | 1.15 | 30/8/2 | 0.30 | 1.26 | 0.209 / 1.33 | 0.225 / 1.17 |
| 2022_02_saudi_arabia | jeddah | 152 | 26 | 0.171 | 0.83 | 19/0/7 | 0.72 | 1.09 | 0.158 / 1.63 | 0.176 / 1.38 |
| 2022_04_emilia_romagna | imola | 192 | 12 | 0.062 | 0.23 | 10/0/2 | 0.52 | 2.48 | 0.119 / 1.92 | 0.178 / 1.34 |
| 2022_06_spain | catalunya | 102 | 38 | 0.373 | 0.67 | 33/5/0 | 0.07 | 0.58 | 0.222 / 1.27 | 0.172 / 1.31 |
| 2022_07_monaco | monaco | 191 | 3 | 0.016 | 0.10 | 0/1/2 | 1.02 | - | 0.019 / 3.60 | 0.171 / 1.31 |
| 2022_08_azerbaijan | baku | 99 | 19 | 0.192 | 0.73 | 16/0/3 | 0.54 | 1.62 | 0.154 / 1.66 | 0.156 / 1.61 |
| 2022_09_canada | villeneuve | 303 | 26 | 0.086 | 0.57 | 7/2/17 | 0.52 | 1.58 | 0.100 / 2.09 | 0.158 / 1.60 |
| 2022_11_austria | red_bull_ring | 290 | 46 | 0.159 | 0.81 | 12/31/3 | 0.15 | 1.20 | 0.131 / 1.82 | 0.116 / 1.79 |
| 2022_12_france | paul_ricard | 198 | 17 | 0.086 | 0.53 | 5/8/4 | 0.34 | 2.39 | 0.105 / 2.04 | 0.150 / 1.53 |
| 2022_17_singapore | marina_bay | 97 | 7 | 0.072 | 0.24 | 3/1/3 | 0.61 | 2.00 | 0.107 / 2.03 | 0.145 / 1.54 |
| 2022_20_mexico | rodriguez | 147 | 19 | 0.129 | 0.31 | 17/2/0 | 0.42 | 1.86 | 0.132 / 1.82 | 0.143 / 1.56 |
| 2022_21_brazil | interlagos | 149 | 42 | 0.282 | 0.96 | 37/3/2 | 0.15 | 1.07 | 0.148 / 1.70 | 0.142 / 1.54 |
| 2022_22_united_arab_emirates | yas_marina | 169 | 42 | 0.249 | 0.96 | 0/38/4 | 0.26 | 1.39 | 0.170 / 1.55 | 0.130 / 1.60 |
| 2023_01_bahrain | sakhir | 122 | 17 | 0.139 | 0.35 | 15/2/0 | 0.05 | 1.11 | 0.209 / 1.33 | 0.238 / 1.10 |
| 2023_07_spain | catalunya | 175 | 48 | 0.274 | 0.77 | 47/1/0 | 0.15 | 0.76 | 0.222 / 1.27 | 0.212 / 1.18 |
| 2024_07_emilia_romagna | imola | 204 | 26 | 0.127 | 0.47 | 24/1/1 | 0.39 | 1.97 | 0.119 / 1.92 | 0.122 / 1.65 |
| 2024_08_monaco | monaco | 378 | 3 | 0.008 | 0.05 | 1/1/1 | 0.51 | 5.12 | 0.019 / 3.60 | 0.049 / 2.38 |
| 2024_12_united_kingdom | silverstone | 193 | 18 | 0.093 | 0.53 | 6/2/10 | 0.21 | 2.10 | 0.095 / 2.14 | 0.100 / 2.17 |
| 2024_21_brazil | interlagos | 187 | 10 | 0.053 | 0.38 | 6/3/1 | 0.18 | 1.90 | 0.148 / 1.70 | 0.203 / 1.42 |
| 2025_23_qatar | lusail | 273 | 4 | 0.015 | 0.12 | 3/1/0 | 0.53 | - | 0.101 / 2.08 | 0.178 / 1.55 |

Where passing happens: mostly in S1, the run from the line to the first braking zone. The
exceptions repeat at the same circuit: Austria (S2, 23/25 and 31/46), Abu Dhabi (S2, 9/10 and
38/42), Canada (S3) and Silverstone (S2/S3). The ghost model works at the line, so the sector
split is descriptive. It cannot improve the line-based rate (the rate is the sum of the
sectors), and it is not used.

**Validation** (`python -m src.sim.overtaking validate`). The question is whether a circuit's
value predicts a race's own passing better than a pooled constant. It is tested over the races
with an earlier (as-of) or another (leave-one-race-out) race at the same circuit:

| target | scope | races | circuit value | pooled constant |
|---|---|---|---|---|
| theta (the race's own fit), mean abs error s | as-of | 12 | 0.59 | 0.71 |
| theta, mean abs error s | leave-one-race-out | 21 | 0.62 | 0.67 |
| rate, mean abs log-odds error | as-of | 13 | 0.80 | 0.88 |
| rate, mean abs log-odds error | leave-one-race-out | 23 | 0.79 | 0.86 |
| rate, binomial log-loss per close lap | as-of | 13 | 0.4215 | 0.4262 |

The circuit factor predicts better, but modestly: the same circuit varies a lot from year to
year (Spain 0.18 to 0.37; Interlagos 0.28, then 0.05 in the wet 2024 race). The biggest miss is
Monaco 2024: the as-of value calls it hard (theta 2.4 s), but its own fit is 5.1 s. The as-of
90% band of the rate covers only 0.62 of the races, so it is too narrow. A public per-race
overtake count was not found in the time available, so there is no second-source check.

**The switch** (`traffic.GhostParams.difficulty`):

- `none`: the earlier rule, the default.
- `circuit`: the circuit's table races, this one included. Judge only, because it reads the
  race (hindsight, ledgered).
- `race`: this race's own count shrunk toward the pooled one. Judge only.
- `asof`: races strictly before this one (`difficulty_prior`; provenance
  `overtaking_difficulty`; the a priori `PRIOR_*` with no earlier race). The belief's.

`circuit` and `race` are refused for the projected source. When the switch is on, a caught car
(not an out-lap ghost) adds the hazard -ln(1 - p) on each lap behind the car ahead, with
p = sigmoid(slope (advantage - theta)), where advantage is the ghost's lap less ours. The
hazard is kept while the car stays within 1.5 s and reset otherwise. Theta is raised by
pass_delta_s - pass_delta_drs_s when DRS is off. The pass completes when the summed hazard
reaches ln 2, the median time to pass. Until then the car is held at the follow gap.

A pass is positional, with no swap, so no time is conjured. The passed ghost is no longer
the car ahead until it is more than 2 s ahead, or until we stop or a neutralisation resets
every fight. `difficulty_dirty_air` takes `dirty_air_s` from the table.

Configs:
- `configs/sim/world_difficulty.yaml`: `world.yaml` plus `difficulty: circuit`.
- `configs/sim/belief_difficulty.yaml`: `belief_traffic.yaml` plus `difficulty: asof`.

The agent's `get_model_report` traffic block adds `passing` when the switch is on: the
difficulty in generic words ("very hard: a faster car usually stays stuck behind for many laps",
"hard", "average", "easy"), the per-lap pass chance when close (90% band), the pace advantage
needed, and where it comes from ("earlier races at this track" etc.). It names no circuit.

**DRS trains** (`difficulty_train`). A follower passes on 4.9% of its close laps when the car
ahead is itself within 1 s of its own car ahead (2,074 laps). It passes on 19.1% when that car
is free (2,878 laps). At the pooled slope the train threshold is 1.8 s higher (`_train_fit`; the
as-of version reads earlier races only). With the switch on, a ghost in a train has theta raised by
shift x (1 - share), and a free ghost has it lowered by shift x share.

**Measured (2026-11-22, calibration races only, nothing adopted).** Base: the adopted default
world (`world_hybrid.yaml`, traffic off). Race-cluster 95% bootstrap over the 19 calibration
races.

Whole-field replay (`src/sim/field_replay.py`, leave-one-driver-out fit, as-of pace, k = 0, 10,
20, 30, 40 pooled), change against the base:

| judge | close-pair order | Spearman | position mae | relative gap s |
|---|---|---|---|---|
| base level | 0.667 | 0.886 | 1.42 | 19.8 |
| traffic (`world.yaml`, one rule) | +0.007 [-0.004, +0.019] | -0.002 | -0.016 | +0.78 [+0.12, +1.31] |
| difficulty `circuit` | -0.001 [-0.013, +0.014] | -0.002 | +0.007 | +0.78 [+0.13, +1.40] |
| difficulty `race` | +0.002 [-0.007, +0.013] | -0.003 | +0.014 | +0.71 [+0.07, +1.31] |
| difficulty `asof` | -0.001 [-0.014, +0.015] | -0.003 | -0.003 | +0.82 [+0.14, +1.48] |
| `circuit` + table dirty air | +0.004 [-0.007, +0.017] | +0.001 | -0.023 | +0.80 [+0.11, +1.46] |
| `circuit` + trains | +0.001 [-0.011, +0.016] | -0.002 | -0.009 | +0.62 [-0.06, +1.24] |
| `asof` + trains | -0.002 [-0.012, +0.011] | -0.001 | -0.008 | +0.54 [-0.12, +1.16] |

By circuit type (judge rate < 0.12 hard: Silverstone, Qatar, Imola, Canada, France; >= 0.18 easy:
Bahrain, Spain; street: Jeddah, Baku, Canada), close-pair change:
- `circuit`: hard -0.006, average +0.000, easy +0.002, street +0.011.
- `circuit` + trains: hard -0.013, average +0.004, easy +0.009, street -0.011.
- One rule: hard -0.011, average +0.015 [+0.002, +0.037], easy +0.014, street -0.004.

Every interval includes 0 except where marked. No variant helps where passing is hard.

Expert consistency on the dev20 scenarios (469, actual_strategy scored on 428), against each
judge's own reference (`python -m src.eval.rerun references|baselines --set dev20`):

| judge | mean regret s | p90 | worst | over 20 s | stint the reference ends 5+ laps earlier | change in mean, hard / street |
|---|---|---|---|---|---|---|
| default (committed dev20) | 7.02 | 19.5 | 69.2 | 0.096 | 0.086 | |
| traffic (`world.yaml`) | 7.35 (+0.33 [+0.10, +0.62]) | 20.5 | 68.9 | 0.112 | 0.070 | +0.06 / +0.31 |
| difficulty `circuit` | 7.36 (+0.34 [+0.09, +0.65]) | 20.6 | 69.0 | 0.110 | 0.075 | +0.25 / +0.23 |
| difficulty `circuit` + trains | 7.37 (+0.35 [+0.10, +0.64]) | 20.4 | 68.9 | 0.110 | 0.091 | +0.41 / +0.22 |
| no traffic, `world_edge: history` | 6.23 (-0.79 [-1.74, +0.02]) | 19.2 | 69.2 | 0.079 | 0.086 | -0.77 / -1.79 [-2.91, -1.19] |

At hard circuits the default judge's mean is 7.66 s; Imola 2024 is worst at 17.8 s, with 36%
over 20 s. With difficulty on, the share of real stints ended 5+ laps early there falls by
-0.034 [-0.101, +0.018] (Imola 2024: 0.24 to 0.08), but the mean regret does not fall.

Traffic in the judge prices the real strategy's own time in traffic as well as the
alternatives', so the real plans' regret rises slightly everywhere. The judge's position
regret for actual_strategy is 0.61 places under both traffic judges.

**Monaco 2024, a diagnostic case** (not in dev20; 25 as-of scenarios sampled into a scratch set
with the `rerun scenarios` builder, digest `dfb8af57c90c`; never committed as a set). It is
the one cached non-held-out race where passing is near impossible and the field ran a managed
pace on HARD for up to 78 laps. Expert consistency of actual_strategy against each judge's
reference (25 scored):

| judge | mean regret s | median | p90 | worst | over 20 s | reference stops / real stops |
|---|---|---|---|---|---|---|
| default (`history_only` edge, no traffic) | 83.6 | 90.2 | 133.8 | 147.6 | 0.88 | 1.16 / 0.32 |
| traffic, one rule | 81.8 | 86.1 | 134.8 | 145.6 | 0.88 | 1.04 / 0.32 |
| difficulty `circuit` | 74.5 | 77.1 | 127.3 | 138.7 | 0.84 | 1.00 / 0.32 |
| difficulty `race` | 75.0 | 77.2 | 128.1 | 138.6 | 0.84 | 1.00 / 0.32 |
| difficulty `asof` | 81.4 | 85.6 | 134.4 | 143.5 | 0.88 | 1.04 / 0.32 |
| `circuit` + trains | 74.3 | 75.3 | 129.5 | 135.4 | 0.80 | 1.00 / 0.32 |
| wear cliff `world_edge: history` | 30.4 | 24.8 | 56.7 | 59.6 | 0.72 | 1.04 / 0.32 |
| wear cliff `world_edge: race` | 30.4 | 24.8 | 56.7 | 59.6 | 0.72 | 1.04 / 0.32 |
| wear cliff off | 30.4 | 24.8 | 56.7 | 59.5 | 0.72 | 1.04 / 0.32 |
| wear cliff off + difficulty `circuit` | 23.2 | 21.3 | 49.8 | 78.7 | 0.52 | 1.00 / 0.32 |

Track position explains a small part: difficulty takes about 9 s off an 84 s regret. Most of the
gap comes from the tyre-life edge. The history-only prior puts the HARD edge far below the 60-78
laps the field really ran, so the judge charges a real 18-lap HARD run to the flag about 90 s
against a stop. The difficulty rule also cannot hold cars at Monaco, because the recorded field
ran several seconds a lap below the model's free-air pace. A car on fresh tyres is then more than
the 3.6 s threshold faster than the ghosts, so it gets by.

**The judge's tyre-life edge** (`wear_cliff.world_edge`, an existing switch; judge only, belief
unchanged). `history` extends the historical edge by the tyre ages this race has laps for
(hindsight, ledgered). On dev20, against `history_only`:
- Expert consistency: mean regret -0.79 [-1.74, +0.02] s (7.02 to 6.23), over 20 s 0.096 to
  0.079, the reference adding a stop the team did not make 0.121 to 0.061. Street circuits
  -1.79 [-2.91, -1.19]; hard -0.77 [-3.17, +1.22]; Imola 2024 17.8 to 12.5 s.
- Whole-field close-pair order: -0.014 [-0.032, -0.002]; relative gap -0.37 [-0.82, +0.02] s.
  `race` gives the same whole-field numbers.
- References with a stint ending beyond the oldest age raced on that compound: 0.198 to 0.299
  (by more than 5 laps: 0.070 to 0.107). A raised edge lets plans run up to it.
- The 25-30-lap SOFT stints at Imola 2024 (SOFT raced 8 laps) appear under both edges, in 72% of
  that race's references either way. They come from the prior being long there, not from the
  extension, so the two findings do not conflict.

Per-car fidelity ablation (`python -m src.sim.fidelity ablate --no-hindsight-variants`, base the
adopted configuration, 19 races, change against the base). `E_real_regret_s` is the new
expert-consistency row: the judge's time for the real plan less its best panel plan, base 5.88 s.

| factor | close pairs | Spearman | rel gap s | B2 rank rho | B2 regret s | pit-now | E_real_regret_s |
|---|---|---|---|---|---|---|---|
| base level | 0.753 | 0.958 | 13.17 | 0.868 | 4.11 | 0.816 | 5.88 |
| world_traffic | -0.046 [-0.090, +0.003] | -0.010 | -0.41 | -0.009 [-0.015, -0.003] | +0.15 | -0.004 | -0.25 [-0.50, -0.02] |
| world_difficulty | -0.052 [-0.105, -0.007] | -0.008 | +0.27 | -0.011 [-0.019, -0.004] | +0.26 [+0.09, +0.43] | -0.003 | -0.24 [-0.47, -0.03] |
| world_difficulty_race | -0.035 [-0.078, +0.006] | -0.014 | +0.11 | -0.012 | +0.26 [+0.06, +0.45] | +0.000 | -0.25 [-0.48, -0.03] |
| belief_traffic | 0 | 0 | 0 | -0.001 | -0.05 | +0.007 | 0 |
| belief_difficulty | 0 | 0 | 0 | -0.005 [-0.008, -0.002] | -0.11 [-0.26, +0.02] | +0.009 | 0 |
| difficulty_both | -0.052 [-0.105, -0.007] | -0.008 | +0.27 | -0.011 | +0.11 | +0.009 | -0.24 |
| cliff_edge_history (no traffic) | -0.022 [-0.067, +0.006] | +0.003 | -0.38 | -0.003 | -0.40 [-0.95, +0.07] | -0.029 [-0.052, -0.007] | -0.96 [-2.46, +0.07] |
| cliff_edge_race (no traffic) | -0.022 [-0.067, +0.006] | +0.003 | -0.38 | -0.004 | -0.09 | -0.017 | -1.49 [-3.04, -0.14] |

E by circuit type, difficulty (circuit): hard -0.14, street -0.02, easy -0.31. Cliff edge
history: hard +0.47, street -0.79, easy -1.11.

**Verdict (selection rule fixed in advance).** No traffic or difficulty variant is an
adopt-candidate:
- The per-car close-pair order falls: -0.046 with one rule, -0.052 [-0.105, -0.007] with the
  circuit difficulty, outside zero.
- The whole-field order does not move.
- Plan-ranking agreement with the belief worsens (B2 rank -0.011, regret +0.26 s).
- Expert consistency moves only in the per-car panel (-0.24 s) and rises slightly against the
  real reference (+0.34 s).
- Every switch stays off.

The DRS-train effect and the table are real findings about passing. They do not make the
judge's order match reality better on these races, where the order error is dominated by pace
error rather than by who is stuck behind whom.

Compute: a dev20 reference build (469 scenarios) costs 0.81 worker-seconds per scenario for the
no-traffic judge, against 4.7-5.4 with any traffic judge on this machine. That is about 180 s
against 1,050-1,220 s per 225 scenarios. The hybrid events make samples differ, so traffic is
recomputed per sample.

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
whose entry is read from `configs/sim/pit_loss_entries.json` (the same entries, precomputed for the
cached races outside the table by `python -m src.sim.pit_loss build-entries`; rebuild it when the
cache or the default fit changes) or else built from its whole race on first use and kept per
process: more cached seasons widen the pool with no code change, and a process loads no race to
price a prior unless the cache holds a race neither file has. A wet race (any INTERMEDIATE or WET lap) is left out of the pool. A held-out race (data/eval_set.json) joins the pool only when the race being priced is
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

**Default: on** (`score.WORLD_EVENTS_DEFAULT` True), chosen by the fidelity scorecard ("The
current configuration" under "Simulator fidelity"): close-pair order +0.016 [0.000, +0.038] and
relative gaps −2.4 s on 18 races, the one factor that improved order and gaps together.
`configs/sim/world_hybrid.yaml` is the default world as a file. The world without SC/VSC:
`score.default_world_config(c, on=False)`, the ablation factor `world_events_off`, or the whole
earlier world `--world-config configs/sim/legacy/world_pre_scorecard.yaml`.

**What the default world models** (`score.WORLD_EVENTS`, events `hybrid`, src/sim/events.py):

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

**No dependence on scoring order.** Every draw is a function of (seed, sample index, stream) only
(`engine.EVENT_STREAM` and the batch streams), and every per-simulator cache built from a snapshot
(the world's and the belief's wear-cliff edges, the windowed fits, the calibrator outputs) is keyed
by the race constants it reads (`engine._race_key`: race, total_laps, compounds). Until the fix
the world's edges were cached once per race from the first snapshot scored; the legacy builder
(`snapshot_at`) gives each driver their own total_laps, so on v1, dev6 and pilot sets a plan's
score depended on the scenarios scored before it (2021 Spain: `fixed_one_stop` up to 1.4 s, the
reference up to 0.1 s). As-of sets (`snapshot_as_of`, the scheduled distance) were not affected:
their references are bit-identical before and after, in either order.
`tests/test_eval_order_independence.py` pins it: alone, after others, reversed, in a second process,
the references in two orders, and the references and baselines stages under `--jobs` 1 and 3.

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

The continuation of STAY_OUT n means: no stop in the next n laps, then the best plan available.
Its plan is the belief's best full plan from the decision snapshot among those whose first stop
is no earlier than the end of lap `laps_completed + n + 1` (a PIT at the reassess snapshot stops
at the end of the next lap), with the same fit, seed (seed + `BELIEF_SEED_OFFSET`), samples,
rules and future-stop limit as `belief_optimizer`. Under SC or VSC at the snapshot it is computed
exactly so (`optimize_batch(snapshot, belief, min_first_laps=n + 1)`, through
`as_continuation(...).stay_plan`): the period in progress runs on per sample as the belief's
event model draws it (the as-of history duration given the laps elapsed), so it is neither
certainly still on nor certainly over at the reassess lap, and nothing is advanced by hand.
Before 2026-11-22 the snapshot was advanced n laps with `track_status` unchanged and the
continuation planned from there, as if the period were certainly still on: it then stopped at
once at the reduced SC/VSC pit loss (2022 Brazil BOT:05, STAY_OUT 10 under SC, was scored as a
stop at the end of lap 17). Under green the snapshot is still advanced n laps and the continuation
planned from there, bit for bit as before: with no period in progress there is no state to carry,
and the two constructions differ only in whether events are drawn over the n stay laps. A PIT is
unchanged. The agent's tools (`evaluate_action`, `compare_plans`, `submit_recommendation`'s check)
expand a stay with the same function, so the plan the model is shown is the plan that is scored.
The tools' comparison plans with a stop at the end of a later lap X onto C (`_stop_at`: the
earliest-legal-stop alternative and `stop_lap_scan`) follow the same rule: under SC or VSC each is
the belief's best full plan from the decision snapshot whose first stop is at the end of lap X onto
C (`optimize_batch(min_first_laps = max_first_laps = X - laps_completed, next_compound=C)`,
`as_continuation(...).stop_plan`); under green, and for a stop now, they are built as before.

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
  (a) `history_only`, the prior's point edge, nothing from this race (the eval world's edge,
  `config.WORLD_EDGE`, chosen by the fidelity scorecard: close pairs +0.025 [0.000, +0.060] against
  b on 18 races, and no hindsight); (b) `history` (the `SimConfig` default and the eval world's
  edge before the scorecard; ablation factor `cliff_edge_history`), the prior extended by the ages
  this race has lap data for, max(prior, oldest age run),
  the judge's one whole-race input here; (c) `race`, the oldest age run in this race (for an
  unraced compound the proxy's practice-age rule), the earlier design. The world's
  `fit_report["support_edge"]` (and `Simulator.cliff_ledger`) records per compound the edge, its
  source ("history:<tier>", "history:<tier> extended by this race's ages to N", "race", ...) and
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
0.44. It was the eval world first; the eval world is now **driver effects alone**
(`driver_effects`: linear_fuel with driver offsets; `config.WORLD_DEGRADATION`,
`score.race_simulators`, `configs/sim/world.yaml`). The fidelity scorecard (`src/sim/fidelity.py`,
six races) finds driver effects the only factor that improves its primary measure (close-pair
finishing order +0.048 [+0.012, +0.077]), the hinge adding nothing and driver effects with the
hinge slightly worse than without (-0.019); the G rule here does not separate the two
(`driver_effects_nonneg` against `driver_effects_hinge`: +0.26 [-0.08, 0.62] s), so it does not
contradict that, and the simpler one is the default. Both are named world overlays for the
scorecard's ablation: `configs/sim/world_de.yaml` and `configs/sim/world_de_hinge.yaml`. The
SimConfig default stays linear_fuel, the comparator the arms are overlays on. Per-compound knots at 75 percent of the longest life (`knot_life_fraction`) and
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

### Thin wear fits: the judge's guard (`wear_guard`, adopted)

Some of the judge's whole-race wear slopes rest on a handful of laps:
- 2022 Brazil HARD: 0.160 s/lap from 10 clean rows of one driver, at ages 2-11.
- 2023 Bahrain MEDIUM: 0.313 from 8 rows.
- 2024 Imola SOFT: -0.015 from 11 rows of two drivers.

An over-steep HARD at Brazil charges a 42-lap HARD stint about 80 s. That is the regret of
run_to_flag / one_stop_total at lap 5 there. A negative SOFT at Imola makes 30-lap SOFT stints
look free.

`SimConfig.wear_guard` (`src/sim/wear_guard.py`) works on the judge's whole-race fit only.
Thresholds were fixed before measuring:
- A compound qualifies when it has at least one, but fewer than `min_drivers` 3 drivers or
  `min_rows` 40 clean race rows.
- Its wear w (fit sd s) is shrunk by precision toward the historical wear p (sd s_p):
  w' = (w/s^2 + p/s_p^2) / (1/s^2 + 1/s_p^2).
- The base moves by (w - w') x the compound's mean fit age, so its mean lap is kept.
- `scale_sd_by_drivers` also widens s by sqrt(3 / drivers) when fewer than 3 cars ran the compound.
- A compound with no race rows is the proxy's and is left alone.

p comes from `data/derived/wear_history.csv` (`python -m src.sim.wear_guard build`): the judge's
own fits (default world, guard off) of 27 non-held-out cached races. `wear_prior` reads it as of
the race:
- strictly earlier races only;
- well-supported fits only (5+ drivers, 60+ rows; no wet race, no 2025 Qatar);
- wear regressed on the compound's common hardness index;
- with fewer than 8 such fits, a priori `ROLE_PRIOR` (sd 0.05).

It is registered in `provenance.py` as `wear_history`. The guard is off in `SimConfig` and left out
of `config_hash` at its default; the eval world turns it on (`config.WORLD_WEAR_GUARD`, adopted
2026-11-22); with it off, the whole-field replay rows are byte-identical to
main.

The fits it changes on the calibration races (wear s/lap, guarded; rows / drivers):

| race | compound | fit (sd) | history (sd) | guard | scaled | rows / drivers |
|---|---|---|---|---|---|---|
| 2023 Bahrain | MEDIUM | 0.313 (0.068) | 0.063 (0.048) | 0.146 | 0.098 | 8 / 1 |
| 2022 Brazil | HARD | 0.160 (0.042) | 0.062 (0.050) | 0.119 | 0.093 | 10 / 1 |
| 2022 Baku | SOFT | 0.129 (0.048) | 0.023 (0.058) | 0.086 | 0.058 | 10 / 1 |
| 2024 Imola | SOFT | -0.015 (0.067) | 0.042 (0.048) | 0.023 | 0.028 | 11 / 2 |
| 2021 Austria | SOFT | 0.042 (0.021) | 0.080 (0.050, a priori) | 0.047 | 0.047 | 36 / 4 |
| 2022 Spain | HARD | 0.166 (0.009) | 0.071 (0.057) | 0.164 | 0.159 | 29 / 1 |
| 2022 Abu Dhabi | SOFT | 0.178 (0.014) | 0.034 (0.048) | 0.167 | 0.167 | 37 / 3 |

**Measured** (2026-11-22, the 19 calibration races, guard against off; base the adopted default;
race-cluster 95%).

Per-car fidelity ablation:

| factor | close pairs | rel gap s | B2 rank rho | B2 regret s | pit-now |
|---|---|---|---|---|---|
| base level | 0.753 | 13.17 | 0.868 | 4.11 | 0.816 |
| `wear_guard` | +0.000 [+0.000, +0.000] | +0.015 [+0.001, +0.043] | +0.009 [+0.001, +0.019] | -0.46 [-1.15, -0.00] | +0.014 [+0.000, +0.039] |
| `wear_guard` + `scale_sd_by_drivers` | +0.000 [+0.000, +0.000] | +0.015 [+0.001, +0.043] | +0.009 [-0.001, +0.021] | -0.56 [-1.31, -0.00] | +0.015 [+0.000, +0.043] |

Whole-field replay (k 0-40): close pairs +0.001 [+0.000, +0.003], relative gap +0.002 s, the
same for both variants.

dev20 rescore of regret against the judge's own reference. Affected = the 7 races in the table
above.

| policy | guard, all | guard, affected | scaled, all | scaled, affected |
|---|---|---|---|---|
| belief_optimizer | -0.34 [-0.97, +0.01] | -0.91 [-2.55, +0.01] | -0.38 [-1.08, +0.01] | -1.03 [-2.82, +0.00] |
| run_to_flag | -0.64 [-1.45, +0.02] | -1.70 [-3.49, -0.26] | -0.81 [-1.83, +0.03] | -2.16 [-4.32, -0.18] |
| one_stop_total | -0.63 [-1.42, +0.02] | -1.69 [-3.46, -0.28] | -0.80 [-1.78, +0.02] | -2.14 [-4.27, -0.26] |
| actual_strategy (expert consistency) | -0.39 [-1.15, +0.00] | -1.01 [-2.84, +0.00] | -0.44 [-1.29, +0.00] | |

- **Brazil 2022 at lap 5** (run_to_flag / one_stop_total on [SOFT 24, HARD 42]): 79.1 s off, 54.9 s
  guarded, 43.2 s scaled.
- **Imola 2024:** actual_strategy 17.8 to 11.6 (11.0 scaled), belief_optimizer -5.7 s. This is
  the negative SOFT wear fixed.
- **Expert consistency at hard circuits:** -1.34 [-3.96, +0.00]. Easy and street circuits are level.
- **References with a stint beyond anything raced:** 0.198 to 0.190.
- **With the guard off,** every dev20 baseline row (plans, times, references, regrets) is identical
  to the committed `reports/eval/rerun/dev20/baselines.csv`.

**Reading under the selection rule.** The order measures hold: per-car close pairs are unchanged
and the whole-field moves +0.001. The relative gap moves +0.015 s. Belief-judge agreement improves:
rank +0.009, the regret of the belief's favourite -0.46 / -0.56 s with intervals excluding 0, and
pit-now +0.014. Expert consistency improves.

Both variants are adopt-candidates for the judge. `scale_sd_by_drivers` is the stronger of the two
pre-stated forms, so it is the recommended one: `wear_guard: {enabled: true, scale_sd_by_drivers: true}`
in the judge's config. Adopted 2026-11-22 as the eval world's default (`config.WORLD_WEAR_GUARD`;
the world before: `configs/sim/legacy/world_pre_v3.yaml`). Every reference and result scored on the
old judge is stale.

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
noise, so the switch was kept; the adopted judge leaves it off ("The current configuration").

Per-race fuel and wear before and after are in the PR description.

### Honest uncertainty

The belief's stated intervals were too narrow. Its fitted factors' 90% intervals covered the judge's
value 55-62% of the time. The residuals it drew as independent lap noise are persistent (lag-1
autocorrelation 0.78-0.94, driver-stint offsets with sd 0.4-1.0 s/lap). Three switches address it,
and the plan_on switch reports the alternative to planning on the draws.

| switch | what it does | default |
|---|---|---|
| `lap_noise: persistent` | each sample draws one offset per plan stint, held over the stint, with sd the as-of driver-stint residual sd, plus the lap-to-lap part iid. Both sds come from the to_snapshot fit's `residual_components` (`degradation._residual_components`: method of moments on the rows the fit saw, so as of the snapshot). The engine draws from its own stream, `LAP_OFFSET_STREAM`, so the other draws are unchanged. | **on** (belief.yaml); `residual` is the belief before |
| `degradation.params.sd_scale` | every sd and the posterior covariance of a kind (base, wear, fuel, hinge) times a factor, S cov S, so the correlations (fuel-wear included) are kept. The point estimate is unchanged. | off |
| `plan_on: point` | `optimize_batch` ranks candidates on the fit's point estimate (`Simulator.point_estimate`: no parameter draw, no lap noise; events and pit-loss draws kept) instead of the mean over draws | off (`draws`) |

The pit-loss sd (as-of `pit_loss`, `PIT_LOSS_STREAM`) and the tyre-life edge's sd (`cliff.sample_ages`,
`CLIFF_STREAM`) were already drawn per sample, so nothing changed for them.

**The width, chosen by coverage.** The factor is the smallest on a grid (1, 1.25, ..., 6) whose 68%
and 90% coverage of the judge's value, pooled over the other races, is within 10 points of nominal.
It is chosen per kind and leave-one-race-out over the 18 calibration races, on the anchored belief.

| kind | before 68 / 90 | LORO factor | held-out 68 / 90 |
|---|---|---|---|
| base | 0.40 / 0.62 | 1.75 | 0.64 / 0.82 |
| fuel | 0.39 / 0.60 | 1.5-1.75 | 0.62 / 0.84 |
| wear | 0.42 / 0.57 | 1.75-2.0 | 0.62 / 0.82 |

The all-race choice is 1.75 for every kind. The tools carry `interval_meaning`, one phrase for what
their 90% ranges mean: "widened so about 9 in 10 such ranges held the full-race value on past races"
when the fit is widened, else "from this race's laps alone; on past races such ranges held the
full-race value far less often".

**Measured** with the fidelity scorecard on the 18 calibration races, against the default judge. All
columns are on the fuel-anchored belief. The first block uses 20 draws and 6 drivers. The second
block uses 60 draws and 1 driver, which reduces the Monte Carlo noise in the replay.

| measure | anchored | + persistent | + sd_scale 1.75 | + both | point estimate |
|---|---|---|---|---|---|
| plan ranking, Spearman | 0.849 | 0.839 | 0.849 | 0.840 | 0.830 |
| judge-regret of the favourite, s | 5.29 | 5.23 | 4.89 | 4.72 | 6.76 |
| pit-now agreement | 0.762 | 0.789 | 0.741 | 0.775 | 0.753 |
| replay close-pair order | 0.640 | 0.635 | 0.635 | 0.629 | 0.653 |
| replay relative gap error, s | 21.72 | 21.66 | 21.91 | 21.90 | 21.93 |
| remaining time 68 / 90 coverage | 0.49 / 0.64 | 0.56 / 0.75 | 0.49 / 0.69 | 0.55 / 0.77 | 0.52 / 0.62 |
| plan-vs-plan gap 68 / 90 coverage | 0.51 / 0.69 | 0.78 / 0.90 | 0.57 / 0.75 | 0.80 / 0.92 | 0.34 / 0.53 |
| replay of real strategies vs reality, 68 / 90 coverage | 0.32 / 0.48 | 0.45 / 0.65 | 0.39 / 0.55 | 0.49 / 0.69 | 0.17 / 0.30 |
| factor 90% coverage, fuel / wear / base | 0.60 / 0.57 / 0.62 | same | 0.85 / 0.80 / 0.82 | 0.85 / 0.80 / 0.82 | same |
| 60 draws: replay close-pair order | 0.639 | 0.636 | 0.630 | 0.631 | |
| 60 draws: replay relative gap error, s | 21.80 | 21.78 | 21.93 | 21.91 | |
| 60 draws: plan-vs-plan gap 68 / 90 coverage | 0.47 / 0.68 | 0.76 / 0.92 | 0.52 / 0.74 | 0.78 / 0.93 | |

**Decisions.**

- **Persistent lap noise is adopted.** It brings plan-versus-plan gap coverage to nominal, raises the
  coverage of remaining time and of the real replay, improves the replay's relative gaps and the
  pit-now agreement, and improves regret.
  - The cost is close-pair order: 2320 of 3650 pair-snapshots against 2332 with 60 draws (2318 against
    2336 with 20), about 0.4 points.
  - Spearman falls by 0.001 with 60 draws and by 0.010 with 20.
  - The judgment call: the gains in coverage, gaps and regret outweigh a loss this small.
- **The parameter widening stays a switch.** It fixes the factor coverage, but it consistently worsens
  both order measures (close-pair -0.009 / -0.010, relative gap +0.13 / +0.19 s). The likely cause
  is the wear bound: wider draws are truncated at 0 more often, and the truncated draws' mean is
  pulled up unevenly across compounds.
  - The tools therefore keep the unwidened factor intervals and say so ("far less often").
- **Planning on the point estimate stays off.** Its replay order is the best of all (0.653, no draw
  noise and no truncation bias), but the planner's ranking agreement and regret are the worst (0.830,
  6.76 s). The draws help ranking here, unlike in the design study (D3, with oracle tyre-life edges).

### Belief priors: where the belief over-stops, and the tyre-life edge at its point

**Where.** On the dev20 development results the belief's plans beat the two fixed rules by about
6 s overall, but by less at circuits with no earlier race in the tyre history, and the rules win
at four low-wear races (2019 Britain, 2021 Austria, 2021 Abu Dhabi, 2022 Azerbaijan). The fidelity
scorecard on the 19 calibration races (adopted configuration at ebde038; 6 drivers, a snapshot
every 5 laps, 20 draws) locates it. Over the plan panel, the belief's favourite has more stops than
the judge's favourite at 14% of snapshots and fewer at 8%. Early in the race (the first third) the
gap is wider: 27% against 8%, a mean of 1.68 stops against 1.49, and a judge-regret of 8.9 s
against 2.4 s mid-race and 1.2 s late. The over-stopping snapshots carry 37% of all the
judge-regret. Their commonest form is two stops where the judge's favourite is one long HARD stint.

**Why.** A diagnosis, not a policy input: at the same snapshots, one belief component at a time
was replaced by the judge's (hindsight), and the judge-regret of the belief's favourite was
measured again (race-cluster 95% intervals):

| belief component replaced by the judge's | regret, all (4.11 s) | early (8.92 s) | no circuit history (4.79 s) | low-wear four (5.19 s) |
|---|---|---|---|---|
| fresh pace and wear of every compound | -2.87 [-3.73, -2.08] | -6.38 [-8.89, -4.12] | -3.19 [-4.10, -2.32] | -3.75 [-4.77, -2.67] |
| the same, HARD only | -0.82 [-1.72, +0.10] | -2.48 [-4.95, -0.62] | -1.19 [-2.42, -0.04] | -1.10 [-2.51, -0.25] |
| fresh pace only (gaps to MEDIUM) | +0.08 [-0.85, +1.09] | -0.55 [-2.37, +1.34] | -0.17 [-1.44, +1.14] | -1.98 [-2.56, -1.32] |
| wear only | +0.30 [-1.45, +2.37] | -0.12 [-3.78, +3.64] | -0.04 [-2.34, +3.12] | +2.77 [-2.46, +8.98] |
| the tyre-life edges (the judge's: the prior's point) | -0.59 [-1.10, -0.18] | -0.53 [-1.02, -0.12] | -0.91 [-1.68, -0.24] | -0.64 [-1.24, -0.12] |
| the edge not drawn (max(seen so far, prior)) | -0.20 [-0.38, -0.06] | -0.44 [-0.92, -0.05] | -0.30 [-0.58, -0.05] | -0.45 [-0.88, -0.11] |
| no parameter draws or lap noise (point estimate) | +0.41 [-0.36, +1.31] | +0.91 [-1.21, +3.23] | +0.51 [-0.74, +1.87] | -0.22 [-0.90, +0.25] |

Most of the regret is the belief's pace and wear per compound early in the race, and only as a
pair: the fitted intercept and slope trade off against each other, so replacing either alone does
nothing. At lap 0 the belief is the practice prior. Practice long runs on a compound come from one
or two cars, in a fit without driver or session terms, and the HARD comes out 0.6 s/lap slower
than MEDIUM fresh against the judge's race fit (2019 Britain: 2.1 s against 0.1 s). At lap 10 the
HARD rows come from the few cars that started on it, often slow ones (Britain 2019: two Haas cars,
whose pace is then taken for the HARD's, +2.4 s). The tyre-life prior is not short on average
against the observed edge (the 0.9 quantile of stint ends: +0.9 to +2.6 laps by compound). The
field's ages seen so far are already a lower bound on the belief's edge, and by lap 20 they rarely
pass it. The edge's spread does matter, though. The cost past the edge is convex, so averaging it
over drawn edges raises the expected cost of every long stint (Jensen's inequality): it penalises
exactly the long one-stop the judge prefers.

**The switch** `belief_cliff_edge` (SimConfig, left out of config_hash at its default): `draw`
(the default, as before) draws each sample's edge, max(seen so far, prior + sd z); `point` takes
max(seen so far, prior) in every sample, the edge the judge's `history_only` prior uses, with the
same lower bound. The tools still report the prior and its sd. Measured on the fidelity scorecard,
point minus draw, race-cluster 95% intervals (* excludes 0):

| split | snapshots | rank rho | regret s | pit-now agreement | over-stop share |
|---|---|---|---|---|---|
| all | 1225 | 0.868, +0.005 [+0.001, +0.010]* | 4.11, -0.20 [-0.38, -0.06]* | 0.816, +0.003 [-0.009, +0.017] | 0.142, -0.008 [-0.019, +0.001] |
| laps 0 to a third | 392 | 0.814, +0.014 [+0.005, +0.028]* | 8.92, -0.44 [-0.92, -0.05]* | 0.719, +0.008 [-0.027, +0.046] | 0.265, -0.010 [-0.037, +0.010] |
| middle third | 461 | 0.892, +0.000 [-0.004, +0.005] | 2.40, -0.15 [-0.42, +0.04] | 0.857, +0.000 [-0.012, +0.014] | 0.124, -0.011 [-0.026, +0.000] |
| last third | 372 | 0.896, +0.000 [+0.000, +0.001]* | 1.16, -0.03 [-0.10, +0.01] | 0.867, +0.003 [-0.005, +0.014] | 0.035, -0.003 [-0.009, +0.000] |
| earlier race at the circuit (8 races) | 499 | 0.892, +0.003 [+0.002, +0.005]* | 3.12, -0.07 [-0.16, -0.00]* | 0.841, +0.004 [+0.000, +0.012] | 0.134, -0.010 [-0.022, +0.000] |
| none (11 races) | 726 | 0.852, +0.006 [+0.000, +0.014]* | 4.79, -0.30 [-0.58, -0.05]* | 0.799, +0.003 [-0.017, +0.025] | 0.147, -0.007 [-0.024, +0.006] |
| the low-wear four | 246 | 0.796, +0.013 [-0.002, +0.034] | 5.19, -0.45 [-0.88, -0.11]* | 0.800, -0.004 [-0.024, +0.013] | 0.130, +0.000 [-0.023, +0.025] |

The judge, the tyre-life component rows and the order measures are unchanged (the judge does not
read the switch; the belief's point edge is the same number). Predictive close-pair order 0.664 ->
0.667; the plans' 90% interval coverage of the judge 0.991 -> 0.988.

The decision-level check: the belief optimizer on the committed dev20 scenarios, re-scored with no
model call (the `baselines` stage into a scratch directory, the references rebuilt at this code;
with the switch off every baseline row is identical to the committed `baselines.csv`). Mean
judge-regret, point minus draw, race-cluster 95% intervals; the rule baselines do not move (they
plan on the point edge already):

| split | scenarios | draw | point | change |
|---|---|---|---|---|
| all | 469 | 3.82 | 3.63 | -0.19 [-0.43, +0.01] |
| laps 0 to a third | 167 | 7.72 | 7.24 | -0.48 [-1.12, +0.06] |
| middle third | 153 | 2.58 | 2.54 | -0.04 [-0.34, +0.25] |
| last third | 149 | 0.71 | 0.70 | -0.01 [-0.05, +0.00] |
| earlier race at the circuit | 194 | 3.35 | 3.25 | -0.10 [-0.36, +0.07] |
| none | 275 | 4.14 | 3.89 | -0.25 [-0.61, +0.07] |
| the low-wear four | 100 | 6.07 | 5.54 | -0.52 [-0.57, -0.47] |

At the low-wear four `run_to_flag` scores 3.22 s, so the rules still win there, by 2.3 s instead of
2.9 s. The switch removes one cause of the over-stopping, not the main one: that is the pace and
wear of the compounds early in the race (the table above), for which no as-of fix tested so far
holds up (2026-11-22 notes, "sim gaps").

**Decision.** An adopt-candidate by the scorecard rule (plan-ranking agreement and judge-regret
improve with race-cluster intervals that exclude 0, pit-now agreement holds). Adopted 2026-11-22:
`belief_cliff_edge: point` in `configs/sim/belief.yaml` and its mirrors (the belief before:
`configs/sim/legacy/belief_pre_v3.yaml`).

### Pre-race publications: circuit ratings and strategy guidance (`SimConfig.prerace`, off)

The tyre supplier publishes two things before each race that a strategist reads and the simulator did
not: the circuit ratings of its race-week preview (traction, braking, lateral, tyre stress, asphalt
grip and abrasion, track evolution, downforce, 1-5) and its strategy guidance after qualifying (the
strategies it expects to be fastest: stops, compounds, sometimes pit-window laps). Both are in
`data/derived/circuit_ratings.csv` and `prerace_guidance.csv` (sources and coverage in
`data/derived/README.md`), keyed by race id and read by `src/sim/prerace.py`. Like the tyre
nominations they are public before the race: `src/sim/provenance.py` registers them as kind
`pre_race` (records no race's data, reads none); every fit that uses them reads earlier races only.

Switches, all off by default (left out of `config_hash` at their defaults); the judge never reads them:

| switch | what it does |
|---|---|
| `prerace.stops_prior_s: L` | the belief's ranking of plans (`optimizer_bridge.optimize_batch`: the look-ahead, `belief_optimizer`, the continuations of a decision, `rank_plans`) adds L s per stop by which a plan's stops left differ from the guidance's (the first expected strategy's stop count less the stops made). Reported times never change; a race with no guidance stop count ranks as before |
| `prerace.tyre_life_guidance: windows\|stated` | the guidance's implied stint lengths (stated pit windows; `stated` also splits the fastest strategy's compound sequence by role life weights 1 : 0.75 : 0.56) combined by precision with the historical tyre-life prior, bias and sd from earlier races |
| `prerace.tyre_life_ratings: replace\|supplement` | the ratings (abrasion, tyre stress, lateral, traction) as covariates of the tyre-life prior, a ridge fit on earlier races, instead of / on top of the nominated-trio severity shift |
| `AGENT_PRERACE=on` | `race_picture` carries `pre_race`: the ratings in words and the expected strategies (no race, circuit, season or driver named), and the v2free prompt one sentence on it |

**Measured** (19 calibration races, main 3ed475f defaults as base; race-cluster 95% intervals; `*`
excludes 0). The stop-count values L = 2 and 5 s were stated before measuring, with the rule "the
larger gain in the first third whose low-wear and overall intervals do not get worse".

Scorecard, belief favourite's judge-regret (s) and over-stop share (favourite has more stops than the
judge's), change against the base (the prior re-ranks the scorecard's plan panel with the same rule):

| split | base regret | L = 2 | L = 5 | base over-stop | L = 2 | L = 5 |
|---|---|---|---|---|---|---|
| all (1225) | 3.36 | -0.27 [-0.50, -0.06]* | -0.43 [-0.84, -0.06]* | 0.140 | -0.021* | -0.028 [-0.067, +0.003] |
| first third (392) | 7.02 | -0.29 [-0.71, +0.03] | -0.58 [-1.18, -0.07]* | 0.263 | -0.036* | -0.048 [-0.112, +0.011] |
| no circuit history (726) | 4.09 | -0.34 [-0.70, -0.01]* | -0.68 [-1.30, -0.10]* | 0.140 | -0.021* | -0.034 |
| low-wear four (246) | 4.62 | +0.00 [-0.00, +0.02] | -0.14 [-0.51, +0.15] | 0.130 | -0.008 | -0.004 |

Same favourite as the judge +0.024* / +0.038*, pit-now agreement +0.020* / +0.026*; plan-rank
correlation is unchanged (the prior moves the favourite, not the belief's times). By the stated rule
L = 5.

dev20 `belief_optimizer` (469 scenarios, the switch in the look-ahead's own search), judge-regret
change against the base: L = 2 -0.19 [-0.50, +0.08], first third -0.33 [-1.10, +0.31], low-wear four
-0.30 [-0.60, +0.00]; L = 5 +0.02 [-0.51, +0.65], first third +0.13 [-1.07, +1.43]. The full search
does not confirm the panel's gain at L = 5: over the whole plan space a strong prior also pulls to
badly timed plans of the expected stop count. Against the rule baselines the look-ahead stays ahead
overall (base: -5.56 s against `run_to_flag`) and behind at the low-wear four (+1.95 base, +1.65 at
L = 2, +1.73 at L = 5).

Live agent (v2free, 100 dev20 scenarios, 70 at laps 0-15; judge regret in s; race:driver cluster
intervals):

| arm | agent | look-ahead | agent - look-ahead |
|---|---|---|---|
| prior off, `AGENT_PRERACE` off | 5.87 | 5.91 | -0.03 [-0.14, +0.05] |
| prior off, block on | 5.91 | 5.91 | +0.00 [-0.10, +0.07] |
| L = 5, block off | 5.92 | 5.72 | +0.20 [-0.02, +0.50] |
| L = 5, block on | 6.03 | 5.72 | +0.32 [-0.11, +0.85] |

Block on minus off: +0.04 [-0.01, +0.08] with the prior off (laps 0-15 +0.075 [+0.025, +0.130],
worse), +0.11 [-0.25, +0.57] with it on. The model cites the block in 1 to 4 departures of 100. The
guidance as text adds nothing measurable, with or without the prior in the look-ahead. With the
prior on, the model gives back the look-ahead's gain: `rank_plans` shows plans faster in time below
the prior-ranked plan 1, and the model departs toward them.

**Decision (2026-11-22).** Not adopted for the rerun. `prerace.stops_prior_s` stays 0 and
`AGENT_PRERACE` stays off. Both remain switches, and the tables stay in the repo. The scorecard's
gain at L = 5 does not carry to the look-ahead's full search on dev20, and the agent gives back
what remains.

Tested and not adopted (no interval excludes 0 in the useful direction):
- tyre-life prior against each race's observed edge (63 observations, as of each race; base MAE 6.96
  laps): ratings replacing the trio shift +0.43 [-0.99, +1.73]; ratings supplementing it -0.03
  [-0.92, +0.87]; guidance windows -0.39 [-1.14, +0.09]; guidance stated -0.24 [-1.16, +0.45];
- the belief's lap-0 and lap-10 wear and HARD/SOFT-vs-MEDIUM gaps against the race's own, corrected by
  an as-of ridge on the ratings, the trio hardness or the track temperature at the snapshot: no
  improvement; ratings worsen the SOFT gap at lap 0 (+0.17 s [+0.02, +0.32]) and track temperature at
  lap 10 (+0.18 [+0.04, +0.35]);
- `tyre_life_guidance: stated` in the scorecard: the lap-0 edge's error against the observed edge
  falls (MAE 5.81 -> 5.62 laps), but the favourite's judge-regret rises: +0.09 [+0.03, +0.17]*, first
  third +0.20 [+0.03, +0.39]*. Rejected.

### Tyre sets at the snapshot (`SimConfig.tyre_inventory`, adopted)

Before this switch every snapshot of the eval and the scorecard carried `sets_available = None`, and
both simulators ran each later stint on a NEW set. In the calibration races, a later stint starts on
a used set:

| compound | later stints on a used set | median age |
|---|---|---|
| SOFT | 66% | 4 laps |
| MEDIUM | 19% | 3 laps |
| HARD | 11% | 2 laps |

Under the inventory below, the belief's favourite needed a new set the driver no longer held at
34.8% of the scorecard snapshots and the judge's at 31.3%; a set of a compound with none left at
0.7% and 0.2%. The tyre sets were adopted in both simulators on 2026-11-22, for the rerun.

**The inventory.** With the switch on, `src/sim/tyre_inventory.py` gives each snapshot the driver's
sets left at that moment, and the existing machinery applies it (`tyre_sets.weekend_sets`,
`rules.inventory`, `Snapshot.sets_available`).

- **Allocation.**
  - From 2020: 2 HARD, 3 MEDIUM and 8 SOFT.
  - On a sprint weekend: 2 HARD, 4 MEDIUM and 6 SOFT.
  - 2019: each driver's allocation was his team's choice and the cache does not hold it. A 2019 race
    therefore takes 13 sets of each compound. It is unconstrained on new sets, but the used sets its
    sessions show are held at their recorded ages.
- **Less:**
  - the sets opened in practice, the sprint and qualifying, each kept at the laps it has run (a set
    used for a few qualifying laps is held at that age);
  - the sets fitted in the race up to the snapshot, from the race laps completed by the end of the
    driver's lap k (`data.visible_rows` on the session times).
- **Approximations, both toward allowing a set:**
  - Sets handed back after practice are not modelled: every opened set is kept as a used set.
  - A used set that matches no held set is not counted against the allocation's unseen new sets,
    because it may be one set counted twice.
- **Information.** Every source is from before the race or is a race lap completed by the snapshot,
  so the inventory is as of the snapshot.
  - The judge reads the same inventory as the belief at each snapshot (ledger entry
    `tyre_inventory`, tag "this race, to snapshot").
  - The judge's whole-race replays run each real stint at its recorded age, as they did before.
  - `tests/test_future_blindness.py` checks that the sets do not move when the race after the moment
    is changed or cut.

**Effect on a plan.**

- **The rule.** A later stint takes the best set of its compound left: a new one while one remains,
  else the least used. A used set starts at its laps run and carries the wear and tyre-life cost of
  that age.
  - A stint naming a used age (a recorded strategy) takes that set, else the nearest held one, or
    refits a set this plan took off earlier at its age then.
  - A plan that needs a compound with no set left is infeasible.
- **Where the rule applies.** The rule is the same in every path a plan takes:
  - the engine (`rules.assign_sets`; `Simulator.compare`'s pit action);
  - the look-ahead and the continuations (`optimizer_bridge.to_contract`: `belief_optimizer`,
    `decisions.to_plan`, `kept_plan`, `named_plan`);
  - `rank_plans` and `evaluate_plan`;
  - the submission check.
- **Where the snapshot is filled.** The snapshot gets its sets where the reference, the policies and
  the scorecard read it (`score.score_scenario`, `reference.compute_reference`,
  `fidelity.section_b`), so what a policy is shown is what is scored.
- **The agent.**
  - `race_picture` gains `sets_left`: per compound, the new sets held and the ages of the used ones.
  - The v2free prompt gains one sentence: sets are limited, and a used set starts at its age.

**The rule baselines.**

| baseline | what it does under the inventory |
|---|---|
| `run_to_flag` | Already keeps only plans that pass every rule on the belief, the inventory included, so it takes its next compound in order. |
| `one_stop_total` | Before the first stop: `fixed_one_stop`. After it: `run_to_flag`. |
| `fixed_one_stop`, `fixed_two_stop`, `greedy_crossover` | Plan as before and take the best set of each named compound. When a compound they name has no set left, they are unavailable for that scenario, with the stated reason "No tyre set for the rule's plan: ...". |
| `actual_strategy` | Replays the recorded strategy with each later stint at its recorded set's age (`data.race_stints`), so the inventory matches the real set, or refits it. A recorded strategy the inventory still cannot fit is unavailable, with the reason. |

**Real strategies against the inventory.** At the race start, the 601 later real stints of the 324
finishers of the 19 calibration races:

- 574 find their exact set;
- 27 (4.5%) take the nearest held set of the compound;
- none is infeasible.

The 27 substitutions:

| where | stints | why |
|---|---|---|
| 2022 Austria (sprint weekend) | 7 MEDIUM | the race fitted a new MEDIUM the estimate no longer holds |
| 2022 Brazil (sprint weekend) | 8 SOFT, 1 MEDIUM | the same: 7 new sets, 2 used ones |
| 2023 Bahrain | 1 SOFT | a new set the estimate no longer holds |
| ten other stints at seven races | 10 | a used set whose recorded age is not held; the nearest held age differs by 1 to 25 laps |

At the two sprint weekends the sessions show more sets than the allocation (a set counted twice in
the timing data), so the new sets left come out short.

One of the ten is 2023 Spain LEC's HARD at 25: a set taken off earlier in the race and refitted. A
plan that took that set off can refit it, so `actual_strategy` plays it exactly.

**Measured on the 19 calibration races, switch off against on** (`python -m src.sim.fidelity run`,
paired on 1225 snapshots, race-cluster 95%; on: the adopted default at 3881500; off: the main of
2026-11-22 with the SC queue):

| measure | off | on | change |
|---|---|---|---|
| judge close pairs | 0.747 | 0.747 | 0 (the judge's whole-race replays are untouched) |
| judge relative gap | 7.60 s | 7.60 s | 0 |
| plan-rank agreement | 0.872 | 0.880 | +0.008 [-0.004, +0.020]; first third +0.015 [+0.002, +0.028] |
| favourite's judge-regret | 3.16 s | 3.47 s | +0.31 [-0.25, +1.09]; first third +0.79 [-0.41, +2.26]; no history +0.56; low-wear four +0.37 |
| pit-now agreement | 0.827 | 0.847 | +0.020 [-0.007, +0.048]; low-wear four +0.033 [+0.013, +0.051] |
| over-stop share | 0.120 | 0.095 | -0.025 [-0.059, +0.001] |
| E_real_regret_s | 5.59 s | 5.21 s | |

**dev20 baselines rescore** (`src.eval.rerun baselines`, references rebuilt; off: the legacy pair,
on: the adopted default at 3881500). Each side's regret is against its own judge's reference.
Change in judge-regret, race-cluster 95%:

| policy | all | first third | low-wear four |
|---|---|---|---|
| `belief_optimizer` | 3.04 -> 3.34 s, +0.30 [-0.32, +1.05] | +0.62 [-0.52, +2.04] | -0.21 [-0.61, +0.06] |
| `run_to_flag` | 9.06 -> 8.28 s, -0.78 [-1.36, -0.29] | -1.13 [-2.01, -0.32] | -0.21 [-0.47, +0.02] |
| `one_stop_total` | 9.82 -> 9.09 s, -0.73 [-1.23, -0.30] | -1.01 [-1.70, -0.42] | -0.19 [-0.47, +0.01] |
| `actual_strategy` | 6.43 -> 6.86 s, +0.43 [-0.26, +1.20] | +0.50 [-0.76, +1.98] | -0.24 [-0.58, +0.12] |
| `fixed_one_stop` | 13.64 -> 14.57 s, +0.93 [-0.30, +2.24] | +0.67 | +1.88 [-0.05, +5.61] |
| `fixed_two_stop` | 22.73 -> 26.49 s, +3.76 [+2.19, +5.62] | +5.45 [+2.51, +8.69] | +2.75 [+0.27, +7.39] |
| `greedy_crossover` (80 rows) | 84.96 -> 111.46 s, +26.5 [+14.1, +42.6] | +29.9 | +19.5 |

Every `actual_strategy` row the off side scores, the on side scores (428 of 469; the 41 others are
unavailable on both sides for the same reasons as before).

The inventory makes two baselines unavailable, both because a named compound has no set left:

- `fixed_one_stop`: 2 more rows ("No tyre set for the rule's plan: ... no HARD set left");
- `fixed_two_stop`: 31 more rows.

References that use a proxied compound: 15 of the 469 scenarios off, 22 on.

**Where the regret change comes from.** One race carries it: 2021 Spain, +414 s of the +381 s total.
Nobody raced HARD there, so both simulators price HARD by proxy. Once the MEDIUMs left are used sets,
the judge moves to the proxied HARD and the belief does not. This is not special-cased (see the
limitation under "Limitations").

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
- the pit window up front: the first message's snapshot, `get_race_snapshot` and
  `get_model_report` carry `pit_window` (`agent_tools.pit_window`: whether pitting now is legal,
  the earliest and latest laps at whose end a stop is legal, in numbers and words), from the
  regulations and the laps so far only;
- a stay-out evaluation is compared with the nearest legal alternative: pitting now when that is
  legal; otherwise a stop at the end of the earliest legal lap (`at_end_of_lap`, with a `note`);
  with no legal stop left, a `note` says STAY_OUT to the flag is the only legal action and there
  is nothing to compare. A stay of k laps is a minimum (the best continuation picks its own stop),
  so every evaluation says what its plan is (`continuation`: "staying out at least k laps; ... the
  first stop is at the end of lap X onto C"), a request that resolves to a plan already evaluated
  in the episode says so in one line (`same_plan_as`), and a stay carries `stop_lap_scan`: up to
  five stop laps from the earliest legal one (over at most 20 laps) onto the alternative's
  compound, each against the evaluated plan under the same draws (mean difference, share faster);
- `submit_recommendation` refuses what the scorer would reject (`agent_tools.submission_error`:
  the decision, a fittable compound for a PIT that is legal now, a whole `reassess_after_laps` of
  at least 1, a confidence in [0, 1], then the scorer's own expansion, `decisions.to_plan`, and the
  rules), with the cause and the legal alternative ("only STAY_OUT to the flag is legal: use
  reassess_after_laps = 3"), so the model corrects it within the episode; anything it accepts the
  scorer scores (tests/test_agent_contract.py);
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
  required key, nothing undeclared) as that call. Since the turn-repeat change it also
  applies the closer repair to the LAST of several envelopes (never an earlier one), and
  reads a submission object that carries an empty `"tool_calls": []` beside an exact
  argument set as that submission (the key dropped, said in the note).
- `AGENT_TURN_REPEAT` (also `run_agent(turn_repeat=...)`) says what an evidence turn whose
  calls are exactly the previous turn's (the same set of tool and arguments; a turn with a
  submission never counts) gets. `run` (the default): the turn runs again like any other
  (`AGENT_REPEAT_MODE` then applies per call). `end`: none of it runs and no budget is
  used; the model is told once that the turn repeats the previous one, with one line per
  call of what it returned (its error whole, so a rule reply's legal alternative; its gap
  reading; or a pointer to the result above) and that a different evidence call or
  `submit_recommendation` moves things forward; a second repeat in a row gets the same in
  other words and the warning that one more ends the evidence phase; the third ends it, and
  the grace turns follow at once. `remind`: the same messages, never the early end. The
  calls of an answered repeat are trajectory events with `repeat_of` and `turn_repeat` (the
  count in the row) and count in `repeat_calls`; a submission after an early end is
  `submitted_in_grace`. Rerun provenance records the mode in effect (the default when the
  variable is unset). Why the switch exists: in the live check of 2026-11-22, two of 18
  episodes re-sent their whole turn eight times with the results in front of them;
  refusing repeated calls (#72) slowed the model and caching them let it loop, since
  neither told it anything new. Why `run` is the default (2026-11-22, was `end`): a clean
  three-way test on one commit, 48 development scenarios per mode, chose by a rule stated
  in advance: fewest wasted calls and tokens without lowering voluntary submissions, regret
  not consulted. `run`: 43 voluntary and 5 grace submissions, 78.5k input tokens, 273
  repeated calls re-run. `remind`: 41 and 7, 92.4k, 193 re-run plus 302 not run. `end`: 32
  and 16, 98.2k, 237 re-run plus 253 not run. Decisions agree on 46 of 48 between every
  pair of modes. Answering or cutting off a repeated turn does not stop the model
  repeating; it makes it use more turns and tokens. `end` and `remind` stay as options.

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

### Agent free choice (v2free, the default)

`AGENT_PROTOCOL` = `v2free` (the default since 2026-11-22) | `v2` (copy plan 1, below) | `v1`.
The agent gets the simulator's look-ahead and the full picture and makes the call; the ranking is
presented as the simulator's best estimate, not the answer.

- **Tools.** v2's (`rank_plans`, `evaluate_plan`, `estimate_tyre_degradation`, `estimate_pit_loss`)
  plus `race_picture` (`agent_tools.race_picture`): position, gaps, laps left, track status and how
  long a neutralisation has lasted, the pit window, the stints so far, the current set and its last
  lap times; the set's observed wear (fuel-corrected lap time on tyre age over its clean laps)
  against the model's value and sd; the cars around (position, gap, compound, tyre age, stops so
  far); the cars within the pit loss behind and the rejoin position a stop now implies, from the
  gaps; per compound the race laps behind the fit, the longest run seen, the field's observed wear
  against the model's, the base pace sd, how a compound with no race laps is priced, and the
  tyre-life estimate with its sd and source. Everything comes from the snapshot and the laps
  visible at its moment (`Simulator.window_rows`, cut by session time), the belief fit and its
  report; the cars are the snapshot's labels. It is a compared surface of the future-blindness
  harness.
- **Any plan.** A submission may name its whole plan (`plan`, [[compound, laps], ...]), which the
  scorer plays exactly (`eval.decisions.named_plan`, its later sets drawn from the inventory as any
  baseline plan's); it must agree with the decision (a PIT's first stint is 1 lap, a STAY_OUT k's at
  least k + 1) and with `compound` and `first_stop_lap` when given, use at most the stop limit and
  modelled compounds, and pass the rules; anything else is refused with the cause. `first_stop_lap`
  and the bare decision work as under v2. v1 and v2 drop a `plan` they were never offered.
- **Prompt.** What the judge scores (the car's race time over the remaining laps); the look-ahead
  can be wrong where it flags a weak estimate or where the race so far contradicts it; the choice is
  any plan; one line of evidence, and when the plan is not plan 1, what in the picture made the
  departure.
- **Checks.** The scorer's own (a submission is accepted only if it can be scored), the named-stop
  and named-plan refusals with the fix and the last-turn fallback, and one contradiction check: a
  submission whose immediate action's best plan the latest `rank_plans` reading puts behind the
  other action's in more than 80% of draws is shown that reading once, and the next submission is
  final. A named plan or first stop that a `rank_plans` or `evaluate_plan` reading put behind plan 1
  in more than 80% of draws gets the same once, with the sign said in words (a positive gap is
  slower). Leaving plan 1 is not by itself held.
- **Candidates.** Under v2free `rank_plans` shows the best plan of each strategy family (stops left
  and compound sequence), at most 8 and none more than 25 s behind plan 1, plus the best PIT-now
  plan, instead of v2's top four shapes. With no model calls over the 469 dev20 scenarios, the
  judge's best plan is among the candidates in 46% (top 4 by time: 46%; laps 0-15: 8% against 5%),
  a plan within 0.5 s of it in 61% (58%; laps 0-15: 19% against 15%), and always picking the
  judge's best candidate would gain 2.19 s against plan 1 (top 4: 1.46 s).

**Headroom over the look-ahead, no model calls (469 dev20 scenarios, SIM_CODE_VERSION 61ec117c6e7b).**
Picking the judge's best of the look-ahead's top 4 with hindsight would cut the judge regret from
3.82 s to 2.35 s (laps 0-15: 7.71 to 5.34; mid: 2.32 to 1.19; late: 0.93 to 0.17), but early in the
race the judge's best plan is among them in 5% of scenarios. No rule a harness could apply found
that room: preferring an unflagged plan within 1 s, 2 s or the 90% gap interval when plan 1 is
flagged changed the regret by +0.003, −0.018 and −0.016 s; preferring the fewest stops within 1 s,
2 s or the interval by −0.021, +0.023 and −0.142 s (cluster CI −0.465, +0.209); and the plan with
the least worst-case gap over the belief's own stress set (pace, wear and tyre-life edge ±1 sd per
compound) cost +0.777 s (+0.446, +1.156). Where a top-4 plan beats plan 1 by the judge by more than
0.5 s (120 scenarios), the robust plan was that plan in 2.

**Development check (103 green dev20 scenarios, the v2 check sample; v2free at 7c31d5b against v2,
--jobs 8 each).**

| | v2 (copy plan 1) | v2free |
| --- | --- | --- |
| valid / voluntary | 103 / 102 | 103 / 102 |
| judge regret mean / median / p90 (s) | 3.66 / 0.52 / 12.58 | 3.78 / 0.52 / 12.93 |
| agent minus belief_optimizer (s, cluster CI) | 0.000 | +0.116 (−0.093, +0.476) |
| plan / action agreement with the optimizer | 103 / 103 | 90 / 102 |
| departures: helped / hurt / same (0.05 s) | 0 | 7 / 3 / 3, net +11.9 s |
| worst departure (s) | 0 | +16.45 |
| calls per episode; tokens in / out | 8.3; 24.8k / 371 | 7.4; 22.7k / 318 |
| latency mean / median (s) | 18.3 / 17.7 | 12.6 / 12.0 |

By stated reason, the check's departures were observed wear against the model 11 (−0.40 s each,
6 helped, 2 hurt), field evidence on a compound 1 (−0.16), and a neutralisation 1 (+16.45: the model
read gap_s +11.1 as faster and confirmed the plan when the check showed it the reading). On the
design sample (100) the same code had 20 departures, 7 helped and 11 hurt (+0.108 s,
−0.009 to +0.285). After the sign wording (d1dee25) the design sample gave +0.058 s (+0.009, +0.127),
19 departures, 4 helped and 12 hurt, and the one +16.45 s check episode, rerun alone afterwards,
kept plan 1. An earlier v2free with v2's top-four candidates (8540ba5) gave +0.040 s (−0.003,
+0.089) on the check, 21 departures, 6 helped and 12 hurt.

### Agent protocol v2

`AGENT_PROTOCOL` = `v2` (the default from #130 until v2free) | `v1` (the prompt, tool surface, loop and
submission handling before it, unchanged; rerun provenance records the protocol in effect). v2 is a
development-side change made after the freeze: the held-out result above was run under v1 and
stands as it is.

- **Tools.** `rank_plans` (no arguments) runs the `belief_optimizer` policy's own search
  (`optimize_batch` on the belief fit, `SEED`, the profile's samples and stop limit) and shows at
  most four distinct plans: the best of each shape (immediate action and compounds), with the best
  PIT-now and the best STAY_OUT plan always shown when both are legal. Plan 1 is the policy's plan.
  Per plan: the stints, the first stop, the immediate action, `submit` (the submission fields that
  keep it), `gap_s` to plan 1 with its 90% interval and `faster_share` over the paired draws,
  `best_share` (fastest of those shown), `first_stop_range` (the stop laps within 1 s for that
  shape) and flags. `decision` is plan 1's immediate action with the best PIT-now plan read against
  the best STAY_OUT plan; `flags` names weak estimates (a compound with no race laps, a
  prior-dominated compound, a safety car or VSC in progress; per plan, a stint past the estimated
  tyre life or past the longest run seen when the life beyond it is a generic prior).
  `evaluate_plan` takes a full plan, or `first_stop_lap` with an optional `compound`, and reads it
  against plan 1. `estimate_tyre_degradation` and `estimate_pit_loss` stay. `evaluate_strategy`,
  `compare_plans`, `get_model_report` and `get_race_snapshot` are v1 only.
- **Submission.** v2 adds an optional `first_stop_lap` to `submit_recommendation`: with `compound`,
  it names the first stop of the plan the submission keeps, and the scorer plays the belief's best
  full plan with that first stop (`eval.decisions.kept_plan`: `optimize_batch` from the snapshot with
  the first stint fixed, the same seed, samples and stop limit as `belief_optimum`). Without it a
  decision is expanded as before. A PIT names the end of the next lap; a STAY_OUT of k laps a stop no
  earlier than the end of lap `laps_completed + k + 1`; anything else is refused with the fix. Under
  v1 the loop drops a `first_stop_lap` the model was never offered.
- **Loop.** Under v2 an argument a tool's schema does not name is refused with the valid ones (v1
  drops it silently), and an unknown tool name gets the tool list. A submission that does not keep
  plan 1 of the model's latest `rank_plans` is held once when a turn is left: the model is shown plan
  1's submit fields and the PIT/STAY_OUT reading, and its next accepted submission is final, changed
  or not (`consistency_check` events in the trajectory; `check_fired`, `check_changed` in the stats).
- **Prompt.** A decision protocol: rank_plans first, plan 1 is the default (copy its submit object),
  a departure only for a plan close to it and only for a named uncertainty flag; one line of
  evidence in the rationale; the 1,024-token reply cap and at most 3 calls per turn. If a named stop
  is refused on the very last turn, the submission is scored without it (`kept_stop_dropped`).

**Why.** The 469 dev20 trajectories at `eval-v2-dev2` (v1): agent minus `belief_optimizer` +0.143 s
(cluster CI +0.025, +0.275), +67 s in all. 33 s came from 45 immediate actions other than the
optimizer's; 35 s from 55 STAY_OUTs that agreed with it but whose `reassess_after_laps` was at
least the laps to its first stop (STAY_OUT k forbids a stop before the end of lap
`laps_completed + k + 1`, which nothing told the model); 8.5 s from 9 PITs onto another compound.
348 episodes swept STAY_OUT k one value at a time, 116 hit the 24-call budget, 132 calls had
invented arguments, 79 turns had no tool call (52 cut off at the reply cap), and in 386 the
confidence was a probability some tool had returned. Mean result sizes: `get_model_report` 6.4k
characters, `compare_plans` 2.9k, `evaluate_strategy` 2.5k (`rank_plans` 2.3k, `evaluate_plan`
0.5k).

**The format's ceiling.** Over all 469 dev20 scenarios (belief side, then the judge's times): the
natural old-format submission for the optimizer's plan (PIT c, or STAY_OUT with k one less than
the laps to its first stop, or to the flag) is scored on that plan in 228 (49%; green 44%, SC 94%,
VSC 100%), and some old-format submission is in 341 (73%). Its judge time differs from the
optimizer's by more than 0.1 s in 45% of scenarios (mean +0.087 s, mean absolute 0.46 s, p95 1.7 s,
max 20.9 s). Naming the first stop scores the optimizer's plan in 469 of 469, judge gap 0.

**Development samples** (seeded, disjoint; scratch lists, not committed). Design: 100 dev20
scenarios, every one where the dev20 agent's immediate action differed from the optimizer's (45),
every SC/VSC one (40), a lap-0 one per race, the rest spread over races. Check: 103 of the
remaining green scenarios, stratified by race and lap. The design sample took four versions (v2a
tools and prompt; v2b the named stop and plan-1 default; v2c the copy instruction and extrapolation
as a note; v2d the last-turn fallback); the check sample ran once, v2c against `origin/main` (v1) as
control, concurrently, `--jobs 8` each, on a loaded machine (load 28-36 on 16 threads), so latency
is comparable between the two only.

| check sample (103) | main (v1) | v2 |
| --- | --- | --- |
| valid / voluntary submissions | 103 / 100 | 102 / 103 (1 named stop never fixed; the fallback above came after) |
| judge regret mean / median / p90 (s) | 3.92 / 0.87 / 13.53 | 3.65 / 0.50 / 12.61 |
| agent minus belief_optimizer (s, cluster CI) | +0.258 (-0.004, +0.558) | 0.000 (0.000, 0.000) on 102 |
| belief_optimizer's own regret (s) | 3.66 | 3.66 |
| scored plan = optimizer's plan | 44 | 102 |
| immediate action = optimizer's | 98 | 102 (+1 invalid) |
| worst departure (s) | +9.98 | 0.00 |
| tool calls / evidence calls per episode | 17.4 / 15.5 | 8.0 / 6.8 |
| repeated calls per episode; budget hits | 3.69; 28 | 0.41; 2 |
| turns with no tool call (cut off) | 17 (4) | 0 |
| input / output tokens per episode | 81.8k / 877 | 23.6k / 364 |
| latency mean / median (s) | 54.0 / 45.3 | 21.6 / 19.6 |

v2 minus main, paired on 102: -0.249 s (cluster CI -0.547, +0.005), 33 better, 48 the same, 21
worse. The invalid episode repeated a `first_stop_lap` without its compound until its turns ran
out; the clearer refusal and the last-turn fallback came after the check run, and that one
scenario rerun once with them (post hoc, not in the table) corrected the compound and scored the
optimizer's plan. On the design sample v2b's model departed from plan 1 eleven times: 8 citing a
flag (+0.016 s each against the optimizer), 3 not (+0.27 s); v2c, with the copy instruction,
departed none of 100 times; v2d (the shipped code) 3 times, all citing a flag (-0.37, -0.37,
+0.45 s), with agent minus optimizer -0.003 s (-0.016, +0.010) and 8.4 calls and 25k input tokens
per episode. So the language model now adds nothing to the look-ahead and takes nothing
away: v2's regret is the optimizer's, and what remains is the predictive simulator's error.

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
  Under the corrected protocol that is `test9`, the nine races of `data/eval_set.json`, built
  once by the rerun pipeline (`scenarios --set test9 --final`, "Rerunning the evaluation").
  The earlier report set, d4, was built and scored once at tag `d4-final-eval` ("The registered
  split" below and "d4, the earlier held-out result (superseded)"); that result is superseded
  ("Result files").

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
fresh 2021-22 draw" below). Nothing loaded them before the final run (test9 at `eval-v2-test`, 2026-11-22): no exploratory commands,
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
  stay in the repo as they are, are not regenerated, and were superseded by d4's (themselves
  superseded now, "Result files").
- **d4** was the report set of the earlier protocol: the four races of that split, 25 scenarios
  each, seed 0, `development: false`. It was built after sprint-weekend support landed (2024
  Austria is a sprint weekend) and scored once at tag `d4-final-eval` ("d4, the earlier held-out
  result (superseded)"). Its 25 Monaco scenario ids are d3's. That result is superseded: it ran
  before the information-contract fixes and under R8 ("Result files"). `data/scenarios/d4.json`
  records the four-race split's `eval_set_digest` (`98edff4380e1...`), not the nine-race file's.
  Like d2 and d3, it is a record of the split it was sampled under, and it still loads. It is
  not regenerated. The report set of the corrected protocol is `test9` (the nine races, built
  by the rerun pipeline).
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
  scenarios per race, seed 0, a development set. Its final pass, at tag `eval-v2-dev2`, is
  committed in `reports/eval/rerun/dev20/` ("Development result (dev20) at eval-v2-dev2").
- `test9`: the nine races of `data/eval_set.json`, 25 per race, seed 0, a report set, tables split
  by era (2019-20, 2021, 2022, 2024-25). The `scenarios` stage refuses a report set without
  `--final`, because building it is the first load of the five new held-out races. It was built
  once, at tag `eval-v2-test` on 2026-11-22, and run once; the run is committed in
  `reports/eval/rerun/test9/` ("Held-out result (test9) at eval-v2-test").

| stage | what it does |
| --- | --- |
| `scenarios` | the set with the leak-free sampler (`--rule as_of`, the default, or `stratified`), builder `calibration.snapshot_as_of`; never over an existing `scenarios.json` |
| `references` | every scenario's reference on the world config, race-parallel (`--jobs`, spawned workers, BLAS at one thread each) and resumable (`parts/references/<race>.pkl`); the cache under `references/` |
| `baselines` | every non-LLM policy (below) on the belief config, race-parallel and resumable (`parts/baselines/<race>.csv`), the references required; `baselines.csv` |
| `agent` | the live agent (`LLM_BACKEND`) on every scenario (or `--only`'s), its tools on the belief config (`AGENT_BELIEF_CONFIG`), `--jobs` episodes at a time (below); `agent.csv`, one trajectory per scenario in `trajectories/`, `agent_run.json`; with `--timing-sample K` first `timing_sample.csv` |
| `tables` | `tables/`: the combined rows, per-race and pooled summary, agent-minus-X paired tables with (race, driver) cluster intervals for every comparator (belief_optimizer stands in when there are no agent rows, named in the file), the chart CSV, the tail table (share over 10, 20, 30 s, median, p90, worst), status counts by reason, by era for a set with eras, position metrics when traffic is on, the agent latency tables when there is a timing sample; and `provenance.json` |
| `smoke` | a live check after a merge: a handful of development scenarios through every stage, then a lint (below) |

**The agent stage in parallel** (`src/eval/agent_pass.py`). Scenarios are independent (one
conversation each, on its own snapshot), so `--jobs N` runs N spawned worker processes that take
scenario ids from one queue in the set's order. Each builds its own simulators (one race at a time,
a race loaded once per worker while its scenarios last), its own backend and policy, BLAS at one
thread, and appends one record per finished scenario to its own `parts/agent/rows_w<k>.jsonl`; the
parent merges the rows in the set's order, so `agent.csv` has the same rows, order and columns for
any N apart from `latency_s` (the default, 1, runs the same code in one process; the only new column
is `api_retries`). Trajectory names are checked to be distinct. `--resume` skips every scenario with
a record and a trajectory (`--rerun-errors` also reruns error rows); a started or finished pass is
never overwritten without `--force`. A failing scenario (an API error the loop reports, an exception
while scoring, a worker that dies) is an `error` row with the message, and the others go on. A 429,
5xx, timeout or dropped connection is retried on the same turn (3 tries, 5 then 20 s apart, the
OpenAI client's own retries off and a 180 s request timeout), counted in `api_retries`; a 401/403
stops the pass, its scenario without a row so that `--resume` runs it. A progress line every
`--progress-every` scenarios (done/total, errors, retries, elapsed, ETA).

**Decision time.** Per-episode `latency_s` in `agent.csv` is measured with N episodes running at
once (and other users of the endpoint), so it is not the decision time. `--timing-sample K` first
runs K scenarios (seeded, stratified across races and early/mid/late laps) one at a time into
`timing_sample.csv` and `timing_sample/trajectories/`; those rows are for timing only and the pass
runs the K again with the rest. The reported decision time (mean, median, p90, with tokens and tool
calls) is the timing sample's; `tables/agent_latency*.csv` pair it with the concurrent pass on the
same K (mean difference with a 95% t interval, ratio, and whether the decision was the same: two
runs of one scenario, a small run-to-run check). `agent_run.json` (copied into `provenance.json`)
records K, N, each pass's start, end, wall time, model and loop switches, and which figure is the
headline.

**Pre-registered reporting** (`configs/eval/reporting.yaml`, `src/eval/reporting.py`). The
reporting choices for the held-out run are fixed in a committed file before the held-out set
exists: the lead comparisons (agent minus `run_to_flag`, then `one_stop_total`), the secondary ones
(`actual_strategy`, `belief_optimizer`), the reference-only and footnoted ones; the statistic (mean
paired regret difference, agent minus comparator, on scenarios both scored) with a 95% percentile
cluster bootstrap on (race, driver) clusters as the primary interval and the race-cluster interval
beside it (2,000 resamples, seed 0) and the verdict words from the primary interval ("ahead",
"behind", "level"); the tail threshold (20 s, with 10 and 30 s beside it, median and worst); the
three-way held-out split in order (the fresh five, the earlier four labelled a second look, all
nine) and by era; the run settings; the handling rules; and the fidelity context printed with the
results. The `scenarios` stage records the file's sha256 in the set's `sampling`; the `tables` stage
refuses a report set whose recorded sha256 differs from the file's, writes
`tables/reporting_paired.csv` (per group, the comparators in order, both intervals, the verdict),
`reporting_tail.csv`, `reporting_counts.csv` (rows per policy and status), `reporting_paired_by_era.csv`
and `reporting_header.txt`, and copies the file verbatim with its sha256 into `provenance.json`. A
development set is one group, "development (all)".

**Smoke run.** `python -m src.eval.rerun smoke [--n 8] [--jobs 4]` (key and `LLM_BACKEND` as for the
agent stage) samples set `smoke` (eight development races; a held-out race is refused, `--final`
is refused) with the leak-free builder, keeps a fixed seeded handful (a lap-0 snapshot, one under
a safety car, one with at most 8 laps left, one mid-race, the rest spread over races), and runs
references, baselines, the agent and the tables for just those under
`reports/eval/rerun/smoke/<UTC time>/` (git-ignored). Then `src/eval/smoke.py` lints them: an
episode that did not submit; a tool result with Python exception text, NaN or inf, a null where a
number is promised or a promised field missing (pit-loss estimate and sd, track status, the
tyre-life estimate with its sd, the gap reading); a string shown to the model naming the real
season, event, circuit or a driver code (the harness's `identity_hits`); a rule reply that does
not say what is legal instead; an error row, an invalid agent row, a negative regret; provenance
without the model id, commit, configs or loop switches. It prints a ten-line summary and one
`FINDING` line each (also `smoke.txt`), and exits 1 on any finding. About 1.5 to 3 minutes at
`--jobs 4`.

`--world-config default` is the scorecard-chosen judge (`score.default_world_config` of
`config.world_default`: driver effects, the history-only tyre-life edge, hybrid SC/VSC) and
`--belief-config default` is `configs/sim/belief.yaml`; the agent's tools run on the scorer's belief
whatever the environment says (`AgentPolicy`, `agent_tools.belief_config_as`), so the tools, the
continuation the scorer expands and `belief_optimizer` plan on one config.
`provenance.json` records the commit (and tag, and whether the tree was dirty), the world and
belief configs with their file sha256, config_hash and each scorecard-decided choice by name
(`rerun.config_choices`: wear fit, driver effects, hinge, wear bound, sd_scale, cliff and its edge,
R8, events, traffic, lap noise, plan_on; for the belief also the calibrator, fuel anchoring and pit
loss), SIM_CODE_VERSION, the set's digest,
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

**Expected exceptions: d3 and d4.** Both are records of the superseded protocol, and half of d3
and all of d4 are held-out races, so neither is rebuilt or rescored. The final evaluation is the
rerun pipeline's `test9` ("Rerunning the evaluation"), which builds its own references.
d3's cached reference (`data/references/d3/9f933a611bea.parquet`, sim_code_version `b62094155bde`)
and d4's (`data/references/d4/39bf1bbb4aa9.parquet`, sim_code_version `58b50a00e0e3`, stale
since the tyre-life change) stay stale on purpose. `reference --check` lists them apart, under
"expected stale (docs/eval.md)" with a reason each, and does not count them as stale
(`reference.EXPECTED_STALE`). The reason strings in the code were written earlier ("d3 is
rebuilt at the final evaluation"; d4 "is re-evaluated in a later, separate step"). The step
they refer to is `test9`: no run rebuilds d3 or d4. `stale_references` still returns both.

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
files below, scored on the old keys, hold for the new ones. `reference --check` then listed only
d3, the expected exception (d4 joined it with the tyre-life change).

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

### Three future stops (a switch, off)

The reference search (`baselines.candidate_plans`, scored by `optimizer_bridge.optimize_batch`)
allows at most `max_stops` future stops; the default is 2 everywhere (the profile's
`sim.max_stops`, `python -m src.eval reference --max-stops`, the fidelity tool's `--max-stops`).
`max_stops: 3` adds every three-stop plan after the plans of up to two stops, so the search
with 2 is unchanged. A candidate list longer than `optimizer_bridge.SCORE_ONE_CALL` (60,000
plans; only a three-stop search reaches it) is scored in chunks of `SCORE_CHUNK` (20,000) under
the same draws, so memory stays bounded.

Measured on the 19 calibration races (main ac85596, the adopted judge, 20 draws, the judge's
lap-0 reference per start set as in A4; no LLM calls, no held-out race):

- The field (A4's field stop counts): 60 of 325 field cars made three or more stops: 17 in 2022
  Bahrain, 14 in 2022 Spain, 8 each in 2023 Bahrain and 2022 Brazil, at most 3 elsewhere. Of
  the full-distance cars, 40 of 209 made three stops and none four; from a snapshot every 5 laps,
  5.0% of (car, lap) pairs had three or more stops still to make, all at laps 0 to 25.
- The judge's best plan with three stops allowed is faster than the reference on 8 of 80 start
  sets: every start set of 2022 Bahrain (3.0 to 4.8 s) and 2022 Spain (0.9 to 5.9 s). On the
  other 17 races the best three-stop plan is 4 to 41 s slower and the reference is unchanged.
  A coarse-to-fine search (stint lengths on a 3-lap grid, then +/-2 laps around the best 30)
  found these plans (per start set: `reports/calibration/three_stops/lap0_references.csv`); the full three-stop search gives the same plan and time on the four start
  sets checked (2022 Bahrain, 2022 Mexico, 2022 Brazil, 2021 Spain).
- Stop count against the field (A4): the lap-0 reference matches the car's own stop count for
  0.471 of the 325 field cars with two stops allowed, 0.548 with three; the race's modal stop
  count 11 of 19 races, 13 of 19 (2022 Bahrain and 2022 Spain, where the field's mode was 3).
- Compute: a full three-stop search scores 0.3 to 1.0 million plans at lap 0 (Bahrain 449,696;
  Brazil 983,320) against at most about 20,000 with two, and took 50 to 121 s per lap-0 reference against
  1 to 1.5 s on the loaded machine (one BLAS thread). Mid-race snapshots have fewer laps left
  and fewer plans.

Decision: left off as a named switch. It changes the reference only where the field really
three-stopped, and it matches the field's stop count better there; it costs about 50 to 90 times
the reference compute. The evaluation's references and results stay on `max_stops` 2.

## The frozen simulator and the evaluation tags

`physics-v1` was the planned name of a simulator release at the end of sprint 2. That tag was
never made, and no result refers to it. The simulators were frozen instead at commit `5f2ec28`
(main after #125, 2026-11-22; SIM_CODE_VERSION `938d268003fa`), with the configuration the
fidelity scorecard chose ("The current configuration (adopted from the scorecard)"). #126 then
finished the agent's tools and committed the pre-registered reporting choices
(`configs/eval/reporting.yaml`). It did not change the simulator: SIM_CODE_VERSION stayed
`938d268003fa`. Its merge is the evaluation commit, tagged `eval-v2-dev`. The only change after
that is #127, under the rule in "Changes after the freeze".

| tag | commit | SIM_CODE_VERSION | what |
| --- | --- | --- | --- |
| `eval-v2-dev` | `6ed9f5e9` | `938d268003fa` | simulators frozen, agent tools final; the first final development pass ran here (superseded, not committed) |
| `eval-v2-dev2` | `9770eb54` | `c8f30186efed` | `eval-v2-dev` plus #127; the development result below |
| `eval-v2-test` | `82f06218` | `c8f30186efed` | `eval-v2-dev2` plus the dev20 results and docs (#128), no code, config or test change; the held-out run |
| `d4-final-eval` | `cd1b937` | `58b50a00e0e3` | the earlier held-out run, under the superseded protocol ("Result files") |

World config hash `5069228b35e7` and belief config hash `351f4124ff22` are the same at every
`eval-v2-*` tag (#127 changed code, not configuration). A run under the rerun pipeline builds its
references in its own directory at the run's SIM_CODE_VERSION, and every row carries the
`sim_code_version`, `world_config_hash` and `belief_config_hash` stamps, which `provenance.json`
lists, so a number cannot come from another simulator version unnoticed.
`data/ground_truth.json` stays the frozen legacy check (`legacy_tools.optimal_strategy`,
`tests/test_case.py`) and is not regenerated from this simulator.

## Held-out result (test9) at eval-v2-test

**The result.** The pre-registration (`configs/eval/reporting.yaml`) makes the fresh five the
primary test: the five races never examined before this run.

- **Fresh five.** The agent is level with both simple rules. Its mean regret is higher than theirs
  by about 1.5 s (+1.59 s against `run_to_flag`, +1.53 s against `one_stop_total`), and both
  intervals include zero. It is also level with the teams' actual strategies, and level with
  `belief_optimizer`, the optimizer its tools run on its own simulator.
- **The development lead does not carry over.** On the calibration races (dev20) the agent was
  about 6 s ahead of the simple rules. On unseen races that lead does not hold.
- **Earlier four.** The agent is ahead of both rules and of the teams. These are the four races
  scored once under the superseded protocol.
- **All nine.** The agent is level with both rules and ahead of the teams.

Everything below is read from `reports/eval/rerun/test9/`.

**What was run, and when.** The one held-out run, on 2026-11-22, at tag `eval-v2-test` (commit
`82f06218`). That commit has the same code as `eval-v2-dev2`; it adds only #128 (results and
docs). The run used a fresh, detached worktree, and the tree was clean throughout
(`provenance.json`: `dirty: false`).

| item | value |
| --- | --- |
| set | `test9` (`configs/eval/rerun_sets.yaml`): the nine races of `data/eval_set.json` (sha256 `4f96f7f3c37a...`), 25 scenarios per race, seed 0, rule `as_of`, `development: false` |
| scenarios | 225, none skipped, digest `663f59b88d5f` |
| scenario file | built once with `--final` at 19:26 UTC and never rebuilt |
| SIM_CODE_VERSION | `c8f30186efed` |
| world | `--world-config default`, config hash `5069228b35e7` |
| belief | `--belief-config default`, config hash `351f4124ff22` |
| reporting choices | `reporting.yaml` sha256 `7bf04ee516bd...`, the same in the file, the set's `sampling` and `provenance.json`; the tables stage checked it |
| model | `qwen3-30b-a3b` on the voyager backend, 1024-token reply cap, the backend's default temperature |
| loop | `max_steps` 10 and `AGENT_TURN_REPEAT=run`; `AGENT_TURN_STATUS`, `AGENT_REPEAT_MODE`, `AGENT_SUBMIT_GRACE_STEPS` and `VOYAGER_TEMPERATURE` checked unset |

The stages ran once each (UTC):

| stage | when | jobs | wall time |
| --- | --- | --- | --- |
| references | 19:29 | 4 | 34 s |
| baselines | 19:30 | 4 | 41 s |
| agent, timing sample (24) | 19:31:48-19:44:17 | 1 | 750 s |
| agent, pass (225) | 19:44:17-20:05:13 | 10 | 1256 s |
| tables `--harness` | after the pass | 1 | 35 s; the future-blindness harness's default part: 38 passed, 7 fixed leaks, 0 known |

These are `agent_run.json` and the run's README in the directory.

**The gate.** Before any model call, the baseline rows had to pass. They did:

- 1,575 rows: seven policies times 225 scenarios, all nine races with 25 each;
- no error and no invalid row, and no negative regret among the 1,348 scored rows;
- `belief_optimizer`, `run_to_flag` and `one_stop_total` scored on every scenario but one.

The other rows are unavailable (a plan that breaks a published rule before it starts:
`tables/status.csv`) or `no_reference`.

**One scenario has no reference.** At `2022_03_australia:ALB:55` the car is on HARD after lap 55,
the tyres 55 laps old, with 3 laps to go and no stop yet made. The row's reason is "No legal
remaining plan exists within the future-stop limit." Every policy's row there is `no_reference`, the agent was not run on it, and it is in no comparison. So
224 scenarios are scored.

**How the committed files were checked.** These checks were done without loading race data,
because the usual commands would load the held-out races again:

- `tables` stage: through `write_provenance`, it computes the as-of pit-loss prior of every race of
  the set (`provenance._pit_loss_for` → `pit_loss.pit_loss_prior`). For a held-out race that pool
  holds the other held-out races, which no committed entry covers, so it would load them through
  `jobs.load_race`.
- lint (`python -m src.eval.smoke`): it builds its identity checker per race with
  `jobs.load_race`.

What was done instead:

- **Tables.** `rerun.stage_tables` was run on a scratch copy of the directory with
  `write_provenance` replaced by a no-op, and with `jobs.load_race` replaced by a function that
  stops the script. It ran to the end, so no race was loaded. All 24 table files came out
  byte-identical to the committed ones, which are the run's originals.
- **Lint.** Every lint check that needs no race data was run on the committed directory:
  - each trajectory's checks without the identity checker (224 trajectories);
  - the row checks;
  - the provenance check.

  The result is 3 findings, the same 3 as the run's own full lint:
  - `ALB:55` "no trajectory";
  - `ALB:55` "no decision on the agent row";
  - "provenance: no model id".
- **The identity check** (no real season, event, circuit or driver code shown to the model) comes
  from the run's own lint, which found no identity hit.

As a follow-up after the poster: the lint's identity checker and the tables stage's provenance
should not need to load race data.

**The model id in provenance.** `provenance.json`'s top-level `model` block is null. The tables
stage fills it from its own environment, and the stage was run without `LLM_BACKEND` and the
model variables set. The model id and the loop settings the pass ran with are recorded in the same
file, under `agent_run.passes[].env`:

- `LLM_BACKEND` voyager;
- `VOYAGER_MODEL` qwen3-30b-a3b;
- `VOYAGER_MAX_TOKENS` 1024;
- `AGENT_TURN_REPEAT` run;
- the other switches null.

That null block is the lint's third finding. It was not fixed: nothing in the run is rerun or
edited.

**Paired, pre-registered** (`tables/reporting_paired.csv`), in the pre-registered order: the fresh
five (the primary test), the earlier four (a second look at the races scored once under the
superseded protocol: 2019 Monaco, 2020 Sakhir, 2024 Austria, 2025 Austria), then all nine.

- The statistic is agent minus X: the mean paired regret difference on the scenarios both scored.
  Negative means the agent is better.
- The primary interval is a 95% percentile cluster bootstrap over (race, driver), with the race
  interval beside it (2,000 resamples, seed 0).
- The verdict comes from the primary interval. "level" means the interval includes zero.
- W / T / L counts the agent's wins, ties and losses.

| group | role | X | paired (clusters) | agent mean | X mean | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fresh five | lead | run_to_flag | 124 (72) | 7.44 | 5.85 | +1.59 | (-0.86, +4.05) | (-5.20, +7.24) | 33 / 40 / 51 | level |
| fresh five | lead | one_stop_total | 124 (72) | 7.44 | 5.91 | +1.53 | (-1.01, +4.08) | (-5.73, +7.50) | 31 / 43 / 50 | level |
| fresh five | secondary | actual_strategy | 116 (69) | 7.28 | 4.95 | +2.33 | (-0.10, +4.64) | (-2.18, +8.50) | 31 / 39 / 46 | level |
| fresh five | secondary | belief_optimizer | 124 (72) | 7.44 | 7.49 | -0.04 | (-0.54, +0.36) | (-0.69, +0.38) | 34 / 61 / 29 | level |
| fresh five | reference only | fixed_one_stop | 121 (71) | 7.63 | 10.87 | -3.24 | (-5.75, -0.68) | (-8.73, +0.95) | 66 / 5 / 50 | ahead |
| fresh five | reference only | fixed_two_stop | 118 (70) | 7.82 | 26.54 | -18.72 | (-21.43, -16.02) | (-21.63, -15.61) | 95 / 0 / 23 | ahead |
| fresh five | footnote | greedy_crossover | 20 (19) | 12.31 | 66.35 | -54.04 | (-85.31, -28.23) | (-82.93, -25.05) | 20 / 0 / 0 | ahead |
| earlier four | lead | run_to_flag | 100 (61) | 5.13 | 10.57 | -5.44 | (-8.56, -2.98) | (-11.91, +0.15) | 50 / 29 / 21 | ahead |
| earlier four | lead | one_stop_total | 100 (61) | 5.13 | 10.51 | -5.38 | (-8.52, -2.92) | (-11.91, +0.21) | 50 / 29 / 21 | ahead |
| earlier four | secondary | actual_strategy | 97 (59) | 4.97 | 17.75 | -12.78 | (-19.33, -7.29) | (-30.38, -2.07) | 51 / 28 / 18 | ahead |
| earlier four | secondary | belief_optimizer | 100 (61) | 5.13 | 5.08 | +0.05 | (-0.19, +0.28) | (-0.13, +0.24) | 25 / 45 / 30 | level |
| earlier four | reference only | fixed_one_stop | 96 (60) | 5.34 | 15.01 | -9.66 | (-12.72, -7.07) | (-20.06, -3.58) | 76 / 0 / 20 | ahead |
| earlier four | reference only | fixed_two_stop | 93 (59) | 5.52 | 27.65 | -22.14 | (-26.70, -18.18) | (-33.62, -13.78) | 85 / 0 / 8 | ahead |
| earlier four | footnote | greedy_crossover | 15 (15) | 10.48 | 81.67 | -71.19 | (-93.97, -47.50) | (-114.71, -38.47) | 15 / 0 / 0 | ahead |
| all nine | lead | run_to_flag | 224 (133) | 6.41 | 7.96 | -1.55 | (-3.50, +0.33) | (-6.53, +3.28) | 83 / 69 / 72 | level |
| all nine | lead | one_stop_total | 224 (133) | 6.41 | 7.97 | -1.56 | (-3.57, +0.36) | (-6.74, +3.40) | 81 / 72 / 71 | level |
| all nine | secondary | actual_strategy | 213 (128) | 6.23 | 10.78 | -4.55 | (-8.08, -1.38) | (-14.70, +2.83) | 82 / 67 / 64 | ahead |
| all nine | secondary | belief_optimizer | 224 (133) | 6.41 | 6.41 | -0.00 | (-0.30, +0.24) | (-0.36, +0.27) | 59 / 106 / 59 | level |
| all nine | reference only | fixed_one_stop | 217 (131) | 6.62 | 12.70 | -6.08 | (-8.05, -4.23) | (-11.79, -1.70) | 142 / 5 / 70 | ahead |
| all nine | reference only | fixed_two_stop | 211 (129) | 6.81 | 27.03 | -20.23 | (-22.64, -17.84) | (-25.64, -16.25) | 180 / 0 / 31 | ahead |
| all nine | footnote | greedy_crossover | 35 (34) | 11.53 | 72.92 | -61.39 | (-81.95, -42.76) | (-85.86, -38.48) | 35 / 0 / 0 | ahead |

On the earlier four the race intervals for the two rules reach zero (+0.15, +0.21): four races
are few clusters. The primary interval decides the verdict.

**By era** (`tables/reporting_paired_by_era.csv`; agent minus X, primary interval, race interval,
verdict):

| era (races) | X | n | agent - X | 95% CI (race, driver) | 95% CI (race) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-20 (2) | run_to_flag | 50 | -6.31 | (-11.74, -1.87) | (-14.78, +2.16) | ahead |
| 2019-20 (2) | one_stop_total | 50 | -6.31 | (-11.74, -1.87) | (-14.78, +2.16) | ahead |
| 2019-20 (2) | actual_strategy | 48 | -23.59 | (-34.94, -13.80) | (-39.29, -6.54) | ahead |
| 2019-20 (2) | belief_optimizer | 50 | +0.14 | (-0.19, +0.50) | (-0.06, +0.34) | level |
| 2021 (2) | run_to_flag | 50 | +0.78 | (-1.32, +2.84) | (+0.69, +0.87) | level |
| 2021 (2) | one_stop_total | 50 | +1.05 | (-1.16, +3.11) | (+0.69, +1.41) | level |
| 2021 (2) | actual_strategy | 50 | -0.11 | (-2.31, +1.82) | (-2.87, +2.65) | level |
| 2021 (2) | belief_optimizer | 50 | +0.25 | (-0.02, +0.55) | (+0.24, +0.26) | level |
| 2022 (3) | run_to_flag | 74 | +2.14 | (-1.73, +6.13) | (-11.02, +11.62) | level |
| 2022 (3) | one_stop_total | 74 | +1.85 | (-2.17, +6.00) | (-12.14, +11.73) | level |
| 2022 (3) | actual_strategy | 66 | +4.18 | (+0.39, +7.83) | (-3.51, +14.82) | behind |
| 2022 (3) | belief_optimizer | 74 | -0.24 | (-1.02, +0.41) | (-1.34, +0.53) | level |
| 2024-25 (2) | run_to_flag | 50 | -4.58 | (-7.05, -2.52) | (-5.88, -3.29) | ahead |
| 2024-25 (2) | one_stop_total | 50 | -4.46 | (-6.91, -2.40) | (-5.64, -3.29) | ahead |
| 2024-25 (2) | actual_strategy | 49 | -2.19 | (-3.38, -0.90) | (-3.68, -0.64) | ahead |
| 2024-25 (2) | belief_optimizer | 50 | -0.05 | (-0.43, +0.20) | (-0.21, +0.12) | level |

A race interval over two races is only the spread of two race means, and the era rows are the
pre-registered breakdown, not a further test.

**Counts** (`tables/reporting_counts.csv`; scored / unavailable / no_reference; there are no
invalid or error rows):

| policy | fresh five | earlier four | all nine |
| --- | --- | --- | --- |
| agent | 124 / 0 / 1 | 100 / 0 / 0 | 224 / 0 / 1 |
| belief_optimizer | 124 / 0 / 1 | 100 / 0 / 0 | 224 / 0 / 1 |
| actual_strategy | 116 / 8 / 1 | 97 / 3 / 0 | 213 / 11 / 1 |
| run_to_flag | 124 / 0 / 1 | 100 / 0 / 0 | 224 / 0 / 1 |
| one_stop_total | 124 / 0 / 1 | 100 / 0 / 0 | 224 / 0 / 1 |
| fixed_one_stop | 121 / 3 / 1 | 96 / 4 / 0 | 217 / 7 / 1 |
| fixed_two_stop | 118 / 6 / 1 | 93 / 7 / 0 | 211 / 13 / 1 |
| greedy_crossover | 20 / 104 / 1 | 15 / 85 / 0 | 35 / 189 / 1 |

**Pooled over all nine** (`tables/chart_regret_s.csv` for the mean and its (race, driver)
interval, `tables/tail.csv` for the rest):

| policy | scored | mean regret s (95% CI) | median | p90 | worst |
| --- | --- | --- | --- | --- | --- |
| agent | 224 | 6.41 (5.27, 7.59) | 1.77 | 22.21 | 28.46 |
| belief_optimizer | 224 | 6.41 (5.22, 7.63) | 1.37 | 22.64 | 30.66 |
| actual_strategy | 213 | 10.78 (7.81, 14.20) | 3.21 | 24.87 | 114.47 |
| run_to_flag | 224 | 7.96 (6.42, 9.62) | 1.53 | 22.05 | 53.99 |
| one_stop_total | 224 | 7.97 (6.38, 9.65) | 1.30 | 23.49 | 53.99 |
| fixed_one_stop | 217 | 12.70 (11.15, 14.19) | 12.14 | 21.93 | 53.99 |
| fixed_two_stop | 211 | 27.03 (25.03, 29.11) | 25.25 | 41.82 | 73.53 |
| greedy_crossover | 35 | 72.92 (53.55, 94.73) | 40.86 | 152.88 | 252.02 |

**Per race**:

- mean regret in seconds, from `tables/summary.csv`, with `actual_strategy`'s scored rows in
  brackets;
- the paired per-race differences, from `tables/combined.csv`: the mean of agent minus
  `run_to_flag`, and of `belief_optimizer` minus `run_to_flag`, on the scenarios both scored;
- the scenarios where the agent's regret equals `belief_optimizer`'s.

| race | group | scored | agent | belief_optimizer | run_to_flag | one_stop_total | actual_strategy | agent - run_to_flag | belief_optimizer - run_to_flag | agent = belief_optimizer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021_07_france | fresh | 25 | 3.97 | 3.74 | 3.10 | 2.56 | 1.32 (25) | +0.87 | +0.63 | 10 of 25 |
| 2021_18_mexico | fresh | 25 | 3.77 | 3.51 | 3.08 | 3.08 | 6.64 (25) | +0.69 | +0.43 | 15 of 25 |
| 2022_03_australia | fresh | 24 | 17.96 | 19.31 | 6.35 | 6.23 | 2.38 (21) | +11.62 | +12.96 | 11 of 24 |
| 2022_05_miami | fresh | 25 | 9.39 | 9.35 | 3.18 | 3.04 | 8.19 (24) | +6.22 | +6.17 | 14 of 25 |
| 2022_14_belgium | fresh | 25 | 2.54 | 2.00 | 13.56 | 14.67 | 6.12 (21) | -11.02 | -11.56 | 11 of 25 |
| 2019_06_monaco | earlier | 25 | 3.47 | 3.13 | 18.25 | 18.25 | 42.75 (25) | -14.78 | -15.12 | 15 of 25 |
| 2020_16_sakhir | earlier | 25 | 3.68 | 3.74 | 1.52 | 1.52 | 10.20 (23) | +2.16 | +2.22 | 9 of 25 |
| 2024_11_austria | earlier | 25 | 10.04 | 10.26 | 13.33 | 13.33 | 10.16 (24) | -3.29 | -3.07 | 11 of 25 |
| 2025_11_austria | earlier | 25 | 3.32 | 3.20 | 9.20 | 8.96 | 7.00 (25) | -5.88 | -5.99 | 10 of 25 |

Over all nine, the agent's regret equals `belief_optimizer`'s on 106 of 224 scenarios.

**Tail** (`tables/reporting_tail.csv`; the stated threshold is 20 s; the five policies the claims
use):

| group | policy | scored | over 20 s (stated) | over 10 s | over 30 s | median s | worst s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fresh five | agent | 124 | 0.194 | 0.266 | 0.000 | 0.71 | 28.46 |
| fresh five | belief_optimizer | 124 | 0.202 | 0.274 | 0.008 | 0.86 | 30.66 |
| fresh five | run_to_flag | 124 | 0.137 | 0.250 | 0.008 | 0.86 | 34.14 |
| fresh five | one_stop_total | 124 | 0.137 | 0.258 | 0.008 | 0.76 | 34.14 |
| fresh five | actual_strategy | 116 | 0.078 | 0.181 | 0.009 | 0.53 | 32.76 |
| earlier four | agent | 100 | 0.040 | 0.170 | 0.000 | 2.20 | 22.64 |
| earlier four | belief_optimizer | 100 | 0.040 | 0.180 | 0.000 | 1.74 | 22.64 |
| earlier four | run_to_flag | 100 | 0.170 | 0.450 | 0.100 | 6.99 | 53.99 |
| earlier four | one_stop_total | 100 | 0.160 | 0.450 | 0.100 | 6.99 | 53.99 |
| earlier four | actual_strategy | 97 | 0.216 | 0.454 | 0.144 | 8.60 | 114.47 |
| all nine | agent | 224 | 0.125 | 0.223 | 0.000 | 1.77 | 28.46 |
| all nine | belief_optimizer | 224 | 0.129 | 0.232 | 0.004 | 1.37 | 30.66 |
| all nine | run_to_flag | 224 | 0.152 | 0.339 | 0.049 | 1.53 | 53.99 |
| all nine | one_stop_total | 224 | 0.147 | 0.344 | 0.049 | 1.30 | 53.99 |
| all nine | actual_strategy | 213 | 0.141 | 0.305 | 0.070 | 3.21 | 114.47 |

On the fresh five, the agent is more than 20 s from the reference on 19.4% of decisions, against
13.7% for either rule. It is never more than 30 s off; the rules sometimes are.

**Development beside held-out** (agent minus X, the primary interval and the verdict;
development from `reports/eval/rerun/dev20/tables/reporting_paired.csv`, labelled development):

| | dev20 (development, 19 calibration races) | fresh five | earlier four | all nine |
| --- | --- | --- | --- | --- |
| agent - run_to_flag | -5.65 (-7.14, -4.23) ahead | +1.59 (-0.86, +4.05) level | -5.44 (-8.56, -2.98) ahead | -1.55 (-3.50, +0.33) level |
| agent - one_stop_total | -6.48 (-8.10, -4.97) ahead | +1.53 (-1.01, +4.08) level | -5.38 (-8.52, -2.92) ahead | -1.56 (-3.57, +0.36) level |
| agent - actual_strategy | -3.18 (-4.55, -1.90) ahead | +2.33 (-0.10, +4.64) level | -12.78 (-19.33, -7.29) ahead | -4.55 (-8.08, -1.38) ahead |
| agent - belief_optimizer | +0.14 (+0.02, +0.28) behind | -0.04 (-0.54, +0.36) level | +0.05 (-0.19, +0.28) level | -0.00 (-0.30, +0.24) level |
| agent mean regret s | 3.96 | 7.44 | 5.13 | 6.41 |
| agent share over 20 s | 0.043 | 0.194 | 0.040 | 0.125 |

**Where the fresh-five shortfall comes from**, as far as the committed files show:

- *Measured.* By race, the agent minus `run_to_flag` is:
  - 2022 Australia +11.62 s;
  - 2022 Miami +6.22 s;
  - 2021 France +0.87 s;
  - 2021 Mexico +0.69 s;
  - 2022 Belgium -11.02 s.

  `belief_optimizer` has the same gap at Australia (+12.96 s) and Miami (+6.17 s), so the language
  model did not add it.
- *Measured.* All five fresh races priced tyre life from the pool alone: tier 3, with no history
  at the same circuit. The tier is `tyre_life.prior_source` in the model reports in the
  trajectories (below).
- *Interpretation.* The run's reading of the trajectories (the README in the directory, cases 2
  and 3) is that in Australia and Miami the belief plans two stops. The judge's best plan there
  is one stop with a long HARD stint, and in the episodes read the tools did not offer that plan.
  Examples:
  - 2022 Miami VER:00: the agent's stay resolves to MEDIUM 16 / SOFT 22 / SOFT 19, against the
    reference MEDIUM 22 / HARD 35;
  - 2022 Australia BOT:10: MEDIUM 15 / MEDIUM 25 / SOFT 8, against MEDIUM 5 / HARD 43.

  So the shortfall sits in the predictive simulator (the belief), probably in its tyre-life and
  wear prior for a circuit it has no history of, and not in the language model. This is a
  reading of a few episodes and the per-race means, not a measured decomposition. No further
  analysis was run, because it would load held-out race data.

**Reliability** (`agent.csv`):

- 224 of 224 scored scenarios submitted a decision: 204 voluntarily and 20 in a grace turn.
- There were 0 invalid rows, 0 error rows and 0 API retries.
- The decisions were 36 PIT and 188 STAY_OUT.
- Grace submissions by race: 2021 France 5, 2024 Austria 5, 2022 Australia 3, 2022 Belgium 3,
  2022 Miami 2, 2020 Sakhir 1, 2021 Mexico 1.
- The run's README reads more from the trajectories:
  - 54 episodes reached the tool-call budget;
  - 42 turns gave no parseable tool call (a reply cut at 1,024 tokens, or JSON with unquoted keys),
    and the loop recovered each one.

**Decision time** (`tables/agent_latency.csv`, `agent_latency_paired.csv`). The isolated timing
sample ran 24 episodes one at a time.

- The machine was quiet: the 1-minute load average had a median of 0.67 (0.19 to 1.17) on 16
  threads, sampled every 30 s (`run_support/load.log`), and no other evaluation or test process
  ran.
- Its mean is **30.2 s** (median 27.8 s, p90 56.7 s). This is the decision-time figure to report.
  It supersedes dev20's 32.9 s, which was an upper bound taken on a loaded machine.

| run | episodes | latency mean / median / p90 s | input / output tokens per episode | tool calls |
| --- | --- | --- | --- | --- |
| isolated (timing sample, 1 job) | 24 | 30.2 / 27.8 / 56.7 | 85.0k / 975 | 19.0 |
| concurrent (`--jobs 10`), the same 24 | 24 | 55.2 / 57.4 / 96.0 | 97.7k / 999 | 19.9 |
| concurrent (`--jobs 10`), all | 224 submitted | 53.5 / 44.3 / 103.3 | 84.5k / 943 | 17.8 |

On the same 24 scenarios, concurrent minus isolated is +25.0 s (95% t interval +13.7 to +36.3),
a ratio of means of 1.83. The decision was the same on 23 of 24 (a small run-to-run check). The
pass took 1,256 s of wall time at 10 jobs, about 5.6 s per episode.

**Tokens and calls** (`agent.csv`, the pass). The mean episode used 84.5k input tokens (median
63.3k) and 943 output tokens (median 797). It made 17.8 tool calls: 15.9 evidence calls, 4.2 of
them repeats. The pass used 18.9 million input and 0.21 million output tokens in all.

**Scenarios under SC or VSC** (`scenarios.json`'s snapshots): 207 green, 14 under a safety car
and 4 under a VSC.

| race | SC | VSC |
| --- | --- | --- |
| 2020 Sakhir | 3 | 0 |
| 2022 Miami | 6 | 2 |
| 2022 Australia | 5 | 2 |

The other six races are all green. The agent's decisions were:

| track status | PIT | STAY_OUT |
| --- | --- | --- |
| green | 33 | 173 |
| SC | 1 | 13 |
| VSC | 2 | 2 |

**The tyre-life prior's history tier per race.** This is `tyre_life.prior_source` in the model
reports the tools returned, counted over every trajectory. The tiers are those of "Historical
tyre-life prior": 1 the same circuit and compound, 3 the pool only, 4 no usable earlier race.

| race | tier |
| --- | --- |
| 2024 Austria, 2025 Austria | 1 (every report) |
| 2020 Sakhir, 2021 France, 2021 Mexico, 2022 Miami, 2022 Australia, 2022 Belgium | 3 (every report) |
| 2019 Monaco | 4, the constant shares (the earliest cached season) |

**Nothing was changed after the held-out set existed.** The scenario file was built at 19:26 UTC
on 2026-11-22. From then on, no file under `src/`, `configs/`, `tests/` or `data/` changed:
`eval-v2-dev2` and `eval-v2-test` have the same code, and only results and docs were committed
since. Nothing in the run was rerun, recomputed in place or edited. The only edit is the
placeholders that replace machine paths in `provenance.json` and `run_support/agent_launch.sh`
(below), and the run's tables are its originals. The defects the run showed are limitations
("Limitations").

**The committed directory.** `reports/eval/rerun/test9/` takes 15.6 MB in 285 files:

- `README.md`: the run's report, as written at the run, with its reading of ten trajectories;
- `scenarios.json`;
- `references/test9/`;
- `baselines.csv`;
- `agent.csv` and `trajectories/`: 224 files, 11.0 MB, uncompressed;
- `timing_sample.csv` and `timing_sample/trajectories/`;
- `agent_run.json`;
- `tables/`: 24 files, the `reporting_*` ones among them;
- `provenance.json`;
- `run_support/`:
  - the agent launch script;
  - `load.log`, the load samples during the run;
  - `make_tables.py`, which wrote the README's tables from `tables/`;
  - `read_traj.py`, the trajectory viewer.

Left out: `parts/` (the intermediates) and the stage logs.

Machine paths were replaced with placeholders, and nothing else was changed:

- in `provenance.json`, the run's scratch output directory became `<out-dir>` in five commands,
  and the belief path's worktree prefix was dropped, leaving `configs/sim/belief.yaml`;
- in `run_support/agent_launch.sh`:
  - the output directory became `<out-dir>`;
  - the worktree became `<worktree at eval-v2-test>`;
  - the main checkout became `<main checkout>`, in the `.env` and Python paths.

## Development result (dev20) at eval-v2-dev2

**These are the races the simulators were calibrated on.** dev20 is a development set: the
nineteen calibration races that are ever sampled. The simulators' fits and choices were made on
these races, so the numbers say how the agent and the baselines behave where the model fits best.
They are not an estimate for unseen races. The held-out result is a separate run on `test9` (the
nine races of `data/eval_set.json`): "Held-out result (test9) at eval-v2-test" above. Nothing in
this section is a held-out number.

**What was run.** At tag `eval-v2-dev2` (commit `9770eb54`, tree clean), the five stages of
"Rerunning the evaluation" over set `dev20`, with these settings:

- the set: 25 scenarios per race, seed 0, rule `as_of`, 469 scenarios (six sampled points of 2022
  Austria are skipped: a set with no recorded compound), digest `a269f07b5b76`;
- the configs: `--world-config default` (config hash `5069228b35e7`) and `--belief-config default`
  (`configs/sim/belief.yaml`, config hash `351f4124ff22`); SIM_CODE_VERSION `c8f30186efed`;
- the reporting choices: `configs/eval/reporting.yaml`, sha256 `7bf04ee516bd...`;
- the model: `qwen3-30b-a3b` on the voyager backend (1024-token reply cap, the backend's default
  temperature);
- the loop: `max_steps` 10 and `AGENT_TURN_REPEAT=run`, with the other loop switches unset;
- the agent stage: an isolated timing sample of 24 scenarios run one at a time, then one pass over
  all 469 at `--jobs 10`.

These are the run settings `reporting.yaml` fixes for the held-out run.

The outputs are committed in `reports/eval/rerun/dev20/`: the scenarios, the references, the
baselines, the agent rows, the trajectories, the timing sample, the tables (the pre-registered
`reporting_*` among them) and `provenance.json`. They take 30 MB in 516 files; the 469
trajectories are 22.6 MB. Its README lists the commands and wall times. The intermediates
(`parts/`) and the logs are not committed. In `provenance.json` the run's scratch output
directory is written `<out-dir>`, and the belief config path is written relative to the project.
Nothing else was edited.

**How the numbers below were computed.** Every number is read from that directory:

- pooled: `tables/chart_regret_s.csv` (mean and its 95% (race, driver) cluster interval),
  `tables/tail.csv` and `tables/reporting_counts.csv`;
- per race: `tables/summary.csv`;
- paired: `tables/reporting_paired.csv` and `tables/reporting_paired_by_era.csv`;
- tail: `tables/reporting_tail.csv`;
- decision time: `tables/agent_latency.csv` and `agent_latency_paired.csv`;
- reliability, decisions and tokens: `agent.csv` (submitted = `decision` set; voluntary = not
  `submitted_in_grace`);
- track status: `scenarios.json`.

The tables are the `tables` stage's output. Running that stage again on a copy of the directory
(`python -m src.eval.rerun tables --set dev20 --out-dir <copy>`, no model calls) regenerates all
14 table files byte-identical. `python -m src.eval.smoke reports/eval/rerun/dev20` (the
trajectory lint) reports 0 findings. The run-to-run estimate also reads the `eval-v2-dev` pass's
`agent.csv`, which is not committed (below).

Fidelity context, from `reporting.yaml`: the judge orders close finishers correctly 0.753 of the
time (ceiling 0.745), and its gap between two cars is off by 13.2 s on average (floor 3.2 s).
Absolute regrets of 4-10 s sit inside that error, so the paired differences carry the claims.

**Pooled** (regret in seconds against the world's reference; 95% interval by (race, driver)
cluster bootstrap):

| policy | scored / unavailable | mean regret s (95% CI) | median | p90 | worst | over 10 s | over 20 s | over 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent | 469 / 0 | 3.96 (3.38, 4.57) | 0.74 | 13.18 | 28.95 | 0.147 | 0.043 | 0.000 |
| belief_optimizer | 469 / 0 | 3.82 (3.24, 4.43) | 0.55 | 13.00 | 29.13 | 0.139 | 0.043 | 0.000 |
| actual_strategy | 428 / 41 | 7.02 (5.73, 8.34) | 2.25 | 19.51 | 69.15 | 0.250 | 0.096 | 0.037 |
| run_to_flag | 469 / 0 | 9.61 (8.34, 10.98) | 4.20 | 24.25 | 79.12 | 0.382 | 0.154 | 0.043 |
| one_stop_total | 469 / 0 | 10.44 (9.04, 11.99) | 4.03 | 25.67 | 79.83 | 0.382 | 0.175 | 0.060 |
| fixed_one_stop | 457 / 12 | 14.81 (13.44, 16.20) | 13.64 | 25.33 | 79.83 | 0.624 | 0.193 | 0.068 |
| fixed_two_stop | 432 / 37 | 24.99 (23.75, 26.22) | 24.43 | 39.61 | 66.63 | 0.894 | 0.674 | 0.343 |
| greedy_crossover | 80 / 389 | 91.91 (78.95, 105.76) | 70.30 | 186.74 | 237.06 | 1.000 | 1.000 | 0.838 |

No row of any policy is invalid, and none has negative regret. The unavailable rows are plans
that break a published rule before they start: `tables/status.csv` gives the reasons. Most are a
stop at once that leaves no first stint; `actual_strategy` also has 13 rows on a compound the
simulator does not price.

**Per race**, mean regret in seconds (`actual_strategy` over the rows it scored, in brackets):

| race | scenarios | agent | belief_optimizer | run_to_flag | one_stop_total | actual_strategy (scored) |
| --- | --- | --- | --- | --- | --- | --- |
| 2019_10_united_kingdom | 25 | 4.76 | 5.00 | 1.99 | 1.99 | 5.14 (23) |
| 2021_01_bahrain | 25 | 4.54 | 4.18 | 13.36 | 13.36 | 8.05 (22) |
| 2021_04_spain | 25 | 2.85 | 2.48 | 9.55 | 14.21 | 7.94 (25) |
| 2021_09_austrian | 25 | 6.35 | 6.50 | 3.12 | 2.70 | 2.85 (25) |
| 2021_20_qatar | 25 | 5.48 | 4.92 | 8.78 | 9.63 | 7.32 (20) |
| 2021_22_united_arab_emirates | 25 | 5.96 | 5.70 | 4.78 | 5.17 | 11.13 (24) |
| 2022_01_bahrain | 25 | 2.09 | 2.08 | 17.73 | 17.73 | 6.15 (22) |
| 2022_02_saudi_arabia | 25 | 1.33 | 1.33 | 8.22 | 7.94 | 4.28 (25) |
| 2022_06_spain | 25 | 1.52 | 0.91 | 20.73 | 21.20 | 13.95 (24) |
| 2022_08_azerbaijan | 25 | 7.20 | 7.08 | 3.00 | 3.06 | 6.11 (25) |
| 2022_09_canada | 25 | 0.74 | 0.74 | 4.29 | 4.29 | 3.83 (24) |
| 2022_11_austria | 19 | 3.57 | 3.40 | 18.08 | 17.53 | 15.68 (4) |
| 2022_12_france | 25 | 3.01 | 3.45 | 6.40 | 6.40 | 3.74 (25) |
| 2022_20_mexico | 25 | 5.06 | 4.95 | 12.14 | 15.04 | 9.57 (22) |
| 2022_21_brazil | 25 | 1.59 | 0.59 | 20.89 | 20.89 | 3.75 (22) |
| 2022_22_united_arab_emirates | 25 | 4.50 | 4.38 | 4.56 | 4.67 | 2.39 (23) |
| 2023_01_bahrain | 25 | 2.00 | 2.17 | 11.08 | 15.82 | 7.43 (24) |
| 2023_07_spain | 25 | 1.65 | 1.61 | 4.46 | 6.90 | 3.16 (24) |
| 2024_07_emilia_romagna | 25 | 10.93 | 10.94 | 11.48 | 11.48 | 17.83 (25) |

**Paired, pre-registered** (`tables/reporting_paired.csv`): agent minus each comparator, the mean
paired regret difference on the scenarios both scored. Negative means the agent is better. The
primary interval is a 95% percentile cluster bootstrap over (race, driver), with the race-cluster
interval beside it (2,000 resamples, seed 0). The verdict comes from the primary interval.
W / T / L counts the agent's wins, ties and losses.

| role | X | paired (clusters) | agent mean | X mean | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lead | run_to_flag | 469 (274) | 3.96 | 9.61 | -5.65 | (-7.14, -4.23) | (-9.02, -2.71) | 226 / 145 / 98 | ahead |
| lead | one_stop_total | 469 (274) | 3.96 | 10.44 | -6.48 | (-8.10, -4.97) | (-9.87, -3.32) | 226 / 145 / 98 | ahead |
| secondary | actual_strategy | 428 (252) | 3.84 | 7.02 | -3.18 | (-4.55, -1.90) | (-4.87, -1.59) | 186 / 156 / 86 | ahead |
| secondary | belief_optimizer | 469 (274) | 3.96 | 3.82 | +0.14 | (+0.02, +0.28) | (-0.002, +0.29) | 102 / 232 / 135 | behind |
| reference only | fixed_one_stop | 457 (273) | 4.06 | 14.81 | -10.74 | (-12.40, -9.15) | (-14.87, -7.12) | 355 / 2 / 100 | ahead |
| reference only | fixed_two_stop | 432 (270) | 4.27 | 24.99 | -20.73 | (-22.17, -19.27) | (-23.46, -17.93) | 392 / 0 / 40 | ahead |
| footnote | greedy_crossover | 80 (74) | 6.75 | 91.91 | -85.16 | (-99.05, -71.98) | (-105.11, -64.13) | 80 / 0 / 0 | ahead |

By era (`tables/reporting_paired_by_era.csv`), agent minus X with the primary interval and the
verdict. The race-cluster interval is not computed for an era with one race.

| era (races) | run_to_flag | one_stop_total | actual_strategy | belief_optimizer |
| --- | --- | --- | --- | --- |
| 2019-20 (1) | +2.77 (+1.04, +4.59) behind | +2.77 (+1.04, +4.59) behind | -0.40 (-4.38, +3.43) level | -0.24 (-0.60, +0.14) level |
| 2021 (5) | -2.88 (-5.19, -0.61) ahead | -3.98 (-7.02, -1.17) ahead | -2.74 (-5.28, -0.33) ahead | +0.28 (+0.02, +0.56) behind |
| 2022 (10) | -8.40 (-10.76, -6.13) ahead | -8.69 (-11.24, -6.30) ahead | -3.19 (-5.30, -1.51) ahead | +0.17 (+0.01, +0.36) behind |
| 2023 (2) | -5.95 (-8.71, -3.58) ahead | -9.54 (-14.25, -5.33) ahead | -3.58 (-7.84, -0.82) ahead | -0.07 (-0.50, +0.28) level |
| 2024-25 (1) | -0.55 (-5.39, +4.03) level | -0.55 (-5.39, +4.03) level | -6.90 (-13.12, -1.15) ahead | -0.01 (-0.24, +0.19) level |

**What the paired table says.** On these races the agent is ahead of both lead rules, by 5.65 s
against `run_to_flag` and 6.48 s against `one_stop_total`. It is also ahead of the teams' actual
strategy, by 3.18 s. Both intervals exclude zero for all three. Per race the rules do better at
4 of the 19: 2019 Britain, 2021 Austria, 2021 Abu Dhabi and 2022 Azerbaijan. The agent is behind
`belief_optimizer`, the optimizer its tools run, by 0.14 s: the primary interval (+0.02, +0.28)
excludes zero, and the race-cluster interval (-0.002, +0.29) just reaches it. The two tie on 232
of 469 scenarios. So on the calibration races the language model adds nothing measurable over its
tools, and costs a little.

**Tail** (`tables/reporting_tail.csv`; the stated threshold is 20 s):

| policy | scored | over 20 s (stated) | over 10 s | over 30 s | median s | worst s |
| --- | --- | --- | --- | --- | --- | --- |
| agent | 469 | 0.043 | 0.147 | 0.000 | 0.74 | 28.95 |
| belief_optimizer | 469 | 0.043 | 0.139 | 0.000 | 0.55 | 29.13 |
| actual_strategy | 428 | 0.096 | 0.250 | 0.037 | 2.25 | 69.15 |
| run_to_flag | 469 | 0.154 | 0.382 | 0.043 | 4.20 | 79.12 |
| one_stop_total | 469 | 0.175 | 0.382 | 0.060 | 4.03 | 79.83 |
| fixed_one_stop | 457 | 0.193 | 0.624 | 0.068 | 13.64 | 79.83 |
| fixed_two_stop | 432 | 0.674 | 0.894 | 0.343 | 24.43 | 66.63 |
| greedy_crossover | 80 | 1.000 | 1.000 | 0.838 | 70.30 | 237.06 |

**Reliability.** All 469 episodes submitted a decision: 439 voluntarily and 30 in a grace turn.
There were 0 invalid rows, 0 error rows and 0 API retries. The decisions were 48 PIT and 421
STAY_OUT. The scenarios were 429 green, 17 under a safety car and 23 under a VSC.

**Decision time.** The isolated timing sample ran 24 episodes one at a time: mean 32.9 s, median
27.6 s, p90 49.4 s. No other evaluation or test process ran during it. Another CPU-heavy process
did, though: a desktop application using about 4.5 cores, with a load average of 5 to 11 on 16
threads. The tools' local computation therefore ran on a loaded machine, so 32.9 s is an upper
bound on the isolated decision time. The concurrent figures, beside it:

| run | episodes | latency mean / median / p90 s | input / output tokens per episode | tool calls |
| --- | --- | --- | --- | --- |
| isolated (timing sample, 1 job) | 24 | 32.9 / 27.6 / 49.4 | 80.5k / 806 | 17.9 |
| concurrent (`--jobs 10`), the same 24 | 24 | 52.9 / 53.2 / 89.0 | 81.5k / 845 | 17.3 |
| concurrent (`--jobs 10`), all | 469 | 54.9 / 48.4 / 98.8 | 83.2k / 919 | 17.6 |

On the same 24 scenarios, concurrent minus isolated is +20.1 s (95% t interval +8.5 to +31.6), a
ratio of means of 1.61. The pass took 2,660 s of wall time for 469 episodes at 10 jobs, about
5.7 s per episode.

**Tokens and calls.** Over the pass the mean episode used 83.2k input and 919 output tokens
(median 57.7k and 789) and 17.6 tool calls: 15.6 evidence calls, of which 4.2 repeated an earlier
call. The pass used 39.0 million input and 0.43 million output tokens in all.

**Run to run.** The `eval-v2-dev` pass ran the same 469 scenarios with the same model and settings,
before #127. #127 changed only what happens under SC or VSC; under green the tools' outputs and the
scoring are byte-identical across the two tags ("Changes after the freeze"). So on the 429 green
scenarios the two passes differ only by the model's run-to-run variation:

- the same PIT / STAY_OUT decision on 407 of 429 (94.9%), and the same scored plan on 311;
- regret, this pass minus the earlier one, +0.035 s per scenario, with a 95% interval of -0.100 to
  +0.192 (`eval.compare.cluster_bootstrap` over (race, driver), 2,000 resamples, seed 0, 264
  clusters).

Under SC and VSC the decision was the same on 17 of 17 and 22 of 23 scenarios, but the tools
changed there, so those are not a run-to-run measure. The timing sample against the pass is a
second, smaller check: the same decision on 22 of 24 (`agent_latency_paired.csv`). The earlier
pass's `agent.csv` is kept with the team's run notes and is not committed. It is superseded.

## Changes after the freeze

**The rule.** The simulators were frozen at `5f2ec28` and the evaluation commit was tagged
`eval-v2-dev`. After that, a change is allowed only if all four of these hold:

1. it fixes a defect with a stated mechanism: what the code does, and why that is wrong;
2. the defect was found on development data;
3. the fix follows from the definition of the behaviour, not from its effect on a result;
4. the recorded decisions are re-scored under the fix, and the re-score is shown: what moved and
   what did not.

Once the held-out scenario file exists, no change of any kind is made. A defect found after that
is reported as a limitation, not fixed.

The rule held. The held-out scenario file was built at 19:26 UTC on 2026-11-22, at `eval-v2-test`.
Nothing under `src/`, `configs/`, `tests/` or `data/` changed after that. The defects the held-out
run showed are under "Limitations".

**Applied.** One change met the rule: PR #127 (`eval-v2-dev` to `eval-v2-dev2`, commits `4efdd11`
and `ab3fd11`).

- *Found* in the `eval-v2-dev` development pass on dev20, in the agent's STAY_OUT decisions under
  SC or VSC (35 rows).
- *Mechanism.* `Simulator.expand_action` turned STAY_OUT k into a plan by advancing the snapshot k
  laps (`engine._advance`), and `_advance` keeps `track_status`. A stay decided under SC or VSC
  therefore planned its continuation as if the neutralisation were certainly still on at the
  reassess lap, so the continuation stopped at once at the reduced SC/VSC pit loss. Two places
  used this expansion:
  - the scorer's expansion of the agent's stays;
  - the tools' stay continuations (`evaluate_action`, `compare_plans`, the submission check), so
    the model was shown the same wrong plans.

  The tools' "stop at the end of lap X" plans had the same carry after the stop (`_stop_at`: the
  earliest-legal-stop alternative and `stop_lap_scan`). References, baselines and
  `belief_optimizer` plan from the snapshot and were not affected.
- *Fix, from the definition.* A policy decides at the snapshot ("Information contract"), where the
  neutralisation's end is unknown. STAY_OUT k means no stop in the next k laps, then the best plan
  available. So a STAY_OUT k under SC or VSC is the belief's best full plan from the decision
  snapshot, with no stop before the end of lap `laps_completed + k + 1`. It uses the same fit,
  seed, samples and rules as `belief_optimizer`, and the period in progress runs on per sample as
  the belief's event model draws it ("Decisions" has the definition). The tools' later-stop plans
  follow the same rule. Green stays and every PIT keep the old path, bit for bit.
  SIM_CODE_VERSION moved from `938d268003fa` to `c8f30186efed`; the config hashes did not change.
- *Three examples* (the reproducer's snapshots, plan before and after):

  | scenario | k | before | after |
  | --- | --- | --- | --- |
  | 2022 Brazil BOT:05 (SC) | 10 | SOFT 12, SOFT 27, HARD 27 | SOFT 20, SOFT 25, MEDIUM 21 |
  | 2022 Azerbaijan VET:10 (VSC) | 2 | HARD 3, MEDIUM 18, MEDIUM 20 | HARD 13, MEDIUM 28 |
  | 2022 Azerbaijan GAS:10 (VSC) | 3 | HARD 4, MEDIUM 19, MEDIUM 18 | HARD 13, MEDIUM 28 |

- *Re-score.* The 469 decisions recorded by the `eval-v2-dev` pass were re-scored at `eval-v2-dev`
  and with the fix, together with every non-LLM policy, with no model calls. Each accepted
  submission was read from its trajectory. Under the fix, 7 rows changed, all of them agent
  STAY_OUT rows under SC or VSC (1 SC and 6 VSC of the 35). The other 28 had a constrained optimum
  equal to the old plan. All 3,283 non-agent rows are bit-identical (references, every baseline,
  `belief_optimizer`), and every regret is still at least 0.

  | scenario | k | agent regret before -> after (s) |
  | --- | --- | --- |
  | 2022_21_brazil:BOT:05 (SC) | 10 | 30.82 -> 1.47 |
  | 2022_08_azerbaijan:VET:10 (VSC) | 1 | 28.26 -> 15.87 |
  | 2022_08_azerbaijan:GAS:10 (VSC) | 10 | 22.56 -> 16.23 |
  | 2021_01_bahrain:HAM:05 (VSC) | 3 | 3.96 -> 7.03 |
  | 2021_01_bahrain:RAI:05 (VSC) | 10 | 4.44 -> 6.21 |
  | 2022_09_canada:HAM:10 (VSC) | 2 | 0.17 -> 0.00 |
  | 2022_09_canada:LAT:10 (VSC) | 1 | 0.03 -> 0.00 |

  | rows | n | agent mean regret before -> after (s) | agent - belief_optimizer before -> after (s) |
  | --- | --- | --- | --- |
  | all | 469 | 4.013 -> 3.921 | +0.197 -> +0.104 |
  | STAY_OUT, green | 383 | 3.872 -> 3.872 | +0.079 -> +0.079 |
  | STAY_OUT, SC | 15 | 3.041 -> 1.085 | +1.941 -> -0.016 |
  | STAY_OUT, VSC | 20 | 3.620 -> 2.917 | +0.704 -> +0.000 |
  | PIT (green 46, SC 2, VSC 3) | 51 | unchanged | unchanged |

  These are decisions the model made while its tools showed the defective plans, so the re-score
  is not a result. The fresh pass at `eval-v2-dev2` is: "Development result (dev20) at
  eval-v2-dev2" above. Its agent minus `belief_optimizer` is +0.14 s, where the re-score gives
  +0.104 and the `eval-v2-dev` pass gave +0.197. Its references and every non-agent row are
  identical to the `eval-v2-dev` pass's apart from the `sim_code_version` stamp.

**Known follow-up, not changed.** Green stays still use the older expansion path: the snapshot
advanced k laps under green, then the continuation planned from there. That path ignores a
neutralisation that starts inside the k laps: the belief's draws of a period starting in those
laps do not enter the plan. The held-out run uses `eval-v2-dev2` as it is. This is listed under
"Limitations".

A PIT now under SC or VSC still plans its continuation from the snapshot one lap on, as before.
That is by definition, not a defect: the in-lap really is run under the neutralisation.

## Result files

**Current: two results, from the same code.**

- `reports/eval/rerun/test9/` is **the held-out result**, at tag `eval-v2-test`: nine races, one
  pass ("Held-out result (test9) at eval-v2-test").
- `reports/eval/rerun/dev20/` is **the development result**, at tag `eval-v2-dev2`, on the
  calibration races ("Development result (dev20) at eval-v2-dev2").

No other committed result is current.

**Everything else is superseded and kept as the record.** No file is deleted.

- Every file in `reports/eval/` outside `rerun/`: the `v1*`, `pilot1*`, `d2*`, `d3*`, `dev6*` and
  `d4*` tables, `case_b/` and `trajectories/`. The table below and the sections after it describe
  them as they were when made. The "current" column means current at that time.
- The tag `d4-final-eval`. It was produced by an evaluation that passed race-wide tyre life (the
  hindsight stint limit R8) and other future information to the policies:
  - the whole-race compound list, lap count, clean-lap filter and pit loss;
  - the snapshot read from the lap after the decision;
  - cross-race tables that held later races.

  See both "Information contract" sections. The corrected protocol ("Rerunning the evaluation")
  replaces it.
- Two development passes on dev20 that were never committed:
  - the provisional pass, before the simulators were frozen;
  - the first final pass at `eval-v2-dev`, before #127 ("Changes after the freeze").

**Superseded by the tyre-life change** (the record of that change). Every result below that was
scored before it was produced under the hindsight stint limit R8 (docs/rules.md R8, retired) and
without the wear cliff:

- the d4 result at tag `d4-final-eval` (`d4_baselines.csv`, `d4_agent.csv`, `d4_combined*.csv`,
  `d4_rule_baselines.csv` and their compare tables);
- the dev6 live agent runs 1-3 (`dev6_agent_pass1*`, `dev6_run2_*`, `dev6_run3_*`);
- the rule-baseline files (`dev6_rule_baselines.csv`, `d4_rule_baselines.csv`,
  `*_combined_rules.csv`, `*_rules_compare_*`);
- the dev6 baselines (`dev6_baselines.csv`, `dev6_run2_baselines.csv`);
- the v1, pilot1 and d2 `*_r8` tables.

`configs/sim/legacy/world_r8.yaml` with `configs/sim/legacy/belief_r8.yaml` reproduces that
setting. `dev6_tyrelife_baselines.csv`, the first row, was the development table after the
change. It is superseded too: it ran before the information-contract fixes, the as-of tables and
the simulator freeze.

| set | current at the time (all superseded now) | superseded earlier (kept as the record) |
| --- | --- | --- |
| test9, every policy and the live agent, at `eval-v2-test` (report set, the nine held-out races; **the current held-out result**) | `reports/eval/rerun/test9/` (scenarios, references, baselines, agent rows, trajectories, timing sample, tables, provenance, the run's README and support files) | d4 (`d4-final-eval`), below |
| dev20, every policy and the live agent, at `eval-v2-dev2` (development, calibration races; **the current development result**) | `reports/eval/rerun/dev20/` (scenarios, references, baselines, agent rows, trajectories, timing sample, tables, provenance) | the provisional and `eval-v2-dev` dev20 passes (not committed) |
| dev6, non-LLM policies under the tyre-life physics (development, a sanity check, not a result to tune against) | `reports/eval/dev6_tyrelife_baselines.csv` (baselines, `belief_optimizer`, `run_to_flag`, `one_stop_total`; world `config.WORLD_DEGRADATION` with the cliff, R8 off; reference rebuilt on this branch) | every R8-era dev6 table below |
| v1, baselines + recorded run | `reports/eval/v1_baselines_r8.csv` (R8 rank 1, margin 3, limit-aware baselines) | `v1_baselines_split.csv` (Milestone 2: belief/world split, before R8); `v1_baselines.csv` (before the split) |
| v1, world traffic on | none under R8 yet | `v1_world_baselines.csv` (before R8) |
| pilot1, baselines | `reports/eval/pilot1_baselines_r8.csv` | `pilot1.csv` baseline rows (Milestone 2, before R8) |
| pilot1, live agent | `pilot1.csv` agent rows (Milestone 2, before R8; the cited ones) | `pilot1_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d2, baselines | `reports/eval/d2_baselines_r8.csv` | `d2.csv` baseline rows (Milestone 2, before R8) |
| d2, live agent | `d2.csv` agent rows (Milestone 2, before R8; the cited ones) | `d2_fix*.csv` is a follow-up rerun after #59, not a replacement (below) |
| d3 | none: its Monaco rows are superseded by d4 (below); d3 is not rescored | `d3_baselines.csv` (Nov 11, before R8; its reference is the expected stale one) |
| d4, baselines + belief_optimizer (report set; the held-out result at tag `d4-final-eval`, **superseded**) | `reports/eval/d4_baselines.csv` (reference `data/references/d4/39bf1bbb4aa9.parquet`) | supersedes d3's 2019 Monaco baseline rows |
| d4, live agent, one pass (report set; the held-out result at tag `d4-final-eval`, **superseded**) | `reports/eval/d4_agent.csv`, trajectories in `reports/eval/trajectories/d4/`; `d4_combined.csv` (agent + baseline + belief_optimizer rows, the input to `d4_compare_regret_s_*` with driver clusters and `d4_compare_race_regret_s_*` with race clusters); per-race paired table `d4_compare_per_race_regret_s_paired.csv`; figure table `d4_chart.csv` | |
| dev6, baselines (development, calibration races) | `reports/eval/dev6_baselines.csv` (R8 rank 1, margin 3, limit-aware baselines) | |
| dev6, live agent pass 1 (development, calibration races) | `reports/eval/dev6_agent_pass1.csv`, trajectories in `reports/eval/trajectories/dev6_pass1/`; `dev6_pass1_combined.csv` (agent + baseline rows, the input to `dev6_pass1_compare_regret_s_*` with driver clusters and `dev6_pass1_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_pass1_compare_per_race_regret_s_paired.csv`; figure table `dev6_pass1_chart.csv` | |
| dev6, baselines + belief_optimizer, run 2 (development, calibration races) | `reports/eval/dev6_run2_baselines.csv` (compound proxy on, reference `fbb33ef732dd`) | `dev6_baselines.csv` (before the proxy) is the record pass 1 was compared with |
| d4, rule baselines, scored once after the d4 results (report set, **superseded**) | `reports/eval/d4_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, frozen at `5a2bc41`); `d4_combined_rules.csv` (`d4_combined.csv`'s rows plus these) and `d4_rules_compare_*` ("Rule baselines added after the held-out run") | `d4_combined.csv` stays as committed |
| dev6, rule baselines (development, calibration races) | `reports/eval/dev6_rule_baselines.csv` (`run_to_flag`, `one_stop_total`, reference `fbb33ef732dd`); `dev6_run2_combined_rules.csv` (run 2's rows plus these) and `dev6_run2_rules_compare_*` ("Rule baselines added after the held-out run") | |
| dev6, live agent run 2 (development, calibration races) | `reports/eval/dev6_run2_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run2/`; `dev6_run2_combined.csv` (agent + baseline + belief_optimizer rows, the input to `dev6_run2_compare_regret_s_*` with driver clusters and `dev6_run2_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run2_compare_per_race_regret_s_paired.csv`; figure table `dev6_run2_chart.csv` | pass 1 stays as the record of the earlier loop and world |
| dev6, live agent run 3 (variance repeat of run 2; development, calibration races) | `reports/eval/dev6_run3_agent.csv`, trajectories in `reports/eval/trajectories/dev6_run3/`; `dev6_run3_combined.csv` (run 3's agent rows + run 2's baseline and belief_optimizer rows, the input to `dev6_run3_compare_regret_s_*` with driver clusters and `dev6_run3_compare_race_regret_s_*` with race clusters); per-race paired table `dev6_run3_compare_per_race_regret_s_paired.csv`; figure table `dev6_run3_chart.csv` | not a replacement: run 2 stays the cited dev6 result; run 3 only measures run-to-run variation |
| dev6, live agent run 4 (loop that logs and recovers from turns with no usable tool call; development, calibration races) | `reports/eval/dev6_run4_agent.csv`, trajectories (with `no_tool_call` events and `parse_repair` notes) in `reports/eval/trajectories/dev6_run4/`; `dev6_run4_combined.csv` (run 4's agent rows + every non-agent row of `dev6_run2_combined_rules.csv`, rule baselines included, the input to `dev6_run4_compare_regret_s_*` with driver clusters and `dev6_run4_compare_race_regret_s_*` with race clusters) | a different agent loop from runs 2 and 3 and from the held-out d4 pass, which all ran before the fix |

Rows scored under R8 carry `stint_limit_rank` and `stint_limit_margin_laps`; rows scored after
the tyre-life change carry a blank `stint_limit_margin_laps` (R8 off).

## v1 numbers under R8 (deterministic; superseded)

Superseded with every file outside `reports/eval/rerun/` ("Result files"); kept as the record.

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

What the held-out run ("Held-out result (test9) at eval-v2-test") showed or rests on:

- **One pass, one model.**
  - The held-out run is one agent pass of one model (`qwen3-30b-a3b`, voyager backend, the
    backend's default temperature). It was not repeated.
  - The run-to-run variation is known only from development data: between two dev20 passes the
    PIT / STAY_OUT decision was the same on 94.9% of green scenarios. The held-out run adds its own
    24 timing-sample pairs, the same decision on 23.
  - Another model, or the same one at another temperature, may land elsewhere.
- **Nine races.**
  - The primary test, the fresh five, is 124 scored scenarios in 5 races. The race-cluster
    intervals there span 12 to 13 s, and one race moves the pooled number (2022 Australia
    +11.6 s, 2022 Belgium -11.0 s against `run_to_flag`).
  - The earlier four had been scored once under the superseded protocol, so they are a second
    look, not a fresh test.
- **The judge's accuracy bounds what a regret means.** The judge (the world simulator) orders
  close finishers correctly 0.753 of the time against a ceiling of 0.745. Its gap between two cars
  is off by 13.2 s on average, against a floor of 3.2 s ("Simulator fidelity"). Mean regrets of
  4-10 s sit inside that error, so only the paired differences carry claims. Even those say how
  the plans compare inside the judge, not on the real track.
- **Default priors where a circuit has no history.**
  - Every fresh race priced tyre life from the pool alone (tier 3), and 2019 Monaco from the
    constant shares (tier 4).
  - The fresh-five shortfall sits in two of those races (2022 Australia and Miami), where the
    belief plans two stops and the judge's best plan is one long HARD stint. `belief_optimizer`
    has the same gap.
  - A better prior for a circuit without history is the likeliest lever. It was not changed (the
    rule below).
- **The model's errors, from the trajectories read** (the run's README, ten episodes read by eye):
  - It misread tool output: 2021 Mexico GIO:00 submitted the plan with p_best 0.02, quoting it as
    0.56.
  - It invented arguments: `stop_lap` on `evaluate_strategy` in 16 calls across 2 episodes, and
    `alternative` in 6 calls in 1 episode (counted over all trajectories). `_dispatch` drops an
    undeclared argument silently, so each call evaluated something other than what the model
    meant, with no message saying so.
  - It gave wrong rationales with a right or neutral decision: a HARD set called "nearing" or
    "past" its longest run at age 19 or 24, and a gap quoted with the wrong sign.
  - It compared different candidate sets and picked within the later one without comparing it to
    the earlier winner: 2022 Belgium MSC:05, 12.8 s worse than `belief_optimizer`.
  - Most of the time the model passes the belief's plan through: its regret equals
    `belief_optimizer`'s on 106 of 224 scenarios, and it is level with it in every group.
- **Nothing was changed after the held-out set existed.** By the rule in "Changes after the
  freeze", the defects above are reported, not fixed, and the run was not repeated.

Standing limitations:

- **Wide, lopsided gap intervals on a prior-dominated compound.** A compound with no race laps
  yet (its pace and wear from practice and the prior) gives gap intervals tens of seconds wide and
  far from symmetric about the median (for example 2021 Spain mid-race, PIT HARD against staying
  out: +26 s, 90% -40 to +96 s). The intervals are the belief's own draws, percentiles not mean
  -+ sd, and the width is the belief saying how little it knows; it is not narrowed.
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
- **A compound nobody raced is priced by proxy in both simulators, and the tyre sets make that
  matter more** ("Tyre sets at the snapshot"). Once the sets left of a raced compound are used, a
  plan moves to a new set of the unraced one, whose price has no race laps behind it in either
  simulator. At 2021 Spain (no car ran HARD) this carries the whole change in the favourite's
  regret when the sets were switched on. It is not special-cased.
- **The tyre sets are estimated, not recorded.** Sets handed back after practice are not modelled
  (every opened set is kept as a used one). 2019 allocations are not in the cache (13 per compound:
  unconstrained on new sets, used sets at their recorded ages). At the two 2022 sprint weekends the
  sessions show more sets than the allocation, so new sets come out short there.
- **v1 scores under a 10-lap minimum stint** (its snapshots predate R2's 3), which makes
  actual_strategy unavailable on 65 scenarios.
- **SC and VSC are modelled, not known.** The belief draws periods from rates and lengths fitted on
  the races before this one; the world (the judge) keeps the period in progress as recorded and
  draws later periods from rates and lengths pooled over the dry races ("Safety cars and VSC in the
  scoring world"). Neither knows when a real later safety car came. Traffic is off on both sides
  by default; position regret has the traffic limits of docs/traffic.md.
- **d3 and d4 are records, not rebuilt.** Their committed references and result files predate
  the tyre-life change (d3's predate R8 too), and both sets include held-out races, so no new
  run is made on them ("Reference cache", "Result files"). The held-out result is the rerun
  pipeline's `test9`.
- **A green STAY_OUT's continuation ignores a neutralisation that starts inside the stay.** A
  STAY_OUT k under green is still expanded by advancing the snapshot k laps and planning from
  there, so the belief's draws of an SC or VSC starting in those k laps do not enter the
  continuation. Under SC or VSC at the snapshot the stay is planned from the snapshot ("Decisions",
  "Changes after the freeze"). Found after the freeze and left as it is.

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
predictive simulator. The ablation's default factors (configs/sim/fidelity/ablation.yaml), on the
current configuration as the base: traffic in the judge (`configs/sim/world.yaml`), in the belief
(`configs/sim/belief_traffic.yaml`) and both; the judge without SC/VSC (`world_events_off`); the
judge's wear fit (driver effects, the default; `world_linear`, neither; `world_hinge`, driver
effects and the hinge); the judge's cliff edge (history only, the default; `cliff_edge_history`,
history extended by this race's ages; `cliff_edge_race`, this race); the cliff off; legacy R8;
the belief without driver effects (`belief_no_driver_effects`) and with the hinge; the wear
bound; the proxy; the belief's pit loss; the trained calibrator. An overlay that names another
events model replaces the events block (`fidelity.merge`), so a hybrid world's `lap_times` does not
reach the rate model. `--field-replay` adds `src/sim/field_replay.py`'s whole-field replay (cars run
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
| noise floor | `ceiling.csv`, `ceiling_pooled.csv` and the scorecard's "Noise floor" section (`fidelity.ceiling_table`, `ceiling_bootstrap`; computed from the cached Section A tables, no simulation): the headline replay (version 1, leave-one-driver-out fit, whole-race pace offset) handed the real race step by step, cumulative: (1) each stint's own pace offset (mean clean-lap residual of the stint, five or more clean laps), (2) in- and out-laps as recorded with no charged pit loss (the real stop durations), (3) SC/VSC laps as recorded, (4) lap 1 as recorded; then (2), (3), (4) alone. Close-pair order and relative gap error per level with race-cluster bootstrap intervals. What level 4 leaves is the non-clean green laps (blue flags, incidents, off-track moments) and lap noise: the order no simulator of these inputs can beat |
| C | the scorecard for the base, each factor overlay alone and `all_on` (configs/sim/fidelity/ablation.yaml); headline changes pooled with a race-cluster bootstrap 95% interval over the races every configuration ran, and the ledger change. A factor whose switch is not on the branch (a module or file it `requires`, or an overlay SimConfig rejects) is skipped with a note |

Before numbers on main (2026-11-22, f7be251: wear cliff, R8 retired, traffic off by default) are in `reports/calibration/fidelity/main/scorecard.md`
and the ablation in `reports/calibration/fidelity/ablation/ablation.md`, both computed on main f7be251
over 18 calibration races (2022_11_austria did not load then).

### The noise floor (9c225f5's default judge, 18 calibration races)

The headline replay handed the real race step by step (`ceiling_pooled.csv`; 365 close pairs;
race-cluster 95% intervals):

| level | close-pair order | rel gap MAE s | Spearman | abs MAE s |
| --- | --- | --- | --- | --- |
| 0 judge (the headline) | 0.682 [0.595, 0.744] | 15.5 [10.1, 21.4] | 0.945 | 32.2 |
| 1 + each stint's own pace | 0.677 [0.581, 0.748] | 15.1 [9.7, 21.1] | 0.946 | 32.5 |
| 2 + in/out laps as recorded (real stop durations) | 0.693 [0.617, 0.753] | 14.5 [9.0, 20.5] | 0.951 | 28.5 |
| 3 + SC/VSC laps as recorded | 0.608 [0.557, 0.667] | 5.2 [4.2, 6.3] | 0.943 | 8.9 |
| 4 + lap 1 as recorded | 0.745 [0.683, 0.804] | 3.2 [2.5, 3.9] | 0.967 | 7.9 |
| pit laps alone | 0.710 [0.635, 0.762] | 14.9 | 0.951 | 28.3 |
| SC/VSC laps alone | 0.551 [0.506, 0.607] | 8.2 | 0.882 | 13.8 |
| lap 1 alone | 0.685 [0.597, 0.749] | 16.0 | 0.946 | 32.6 |

Reading: about 0.75 close-pair order and 3 s relative gap error is the ceiling for a judge that
knows every stint's pace, every stop, every neutralised lap and the start; what is left sits on
the non-clean green laps (blue flags, incidents, off-track moments). Most of today's relative gap
error is the neutralised laps (15.5 to 5.2 s), not tyre or pace physics: per-stint pace known
moves the headline by -0.005 order and -0.4 s, the real stops by +0.011 to +0.028 order and
-0.6 to -1.0 s. Recorded SC/VSC laps without the cars' real positions at the restart make the
order worse (0.55 alone): a recorded SC lap closes the real gap to the car ahead, not the
simulated one, so the neutralised laps need the queue (whose order the earlier laps decide)
rather than their lap times.

### Pit-stop durations

Switch `SimConfig.pit_stops` (`replay`, `team`, `team_prior_stops`), off by default. The judge
charges every stop one whole-race pit loss (the median green stop), but real stops vary by
seconds (pit-lane time, entry to exit: median absolute deviation about 0.5 s on the calibration
races, sd 1.4 to 4.7 s with the slow ones, team medians spread 0.6 to 2.5 s) and a slow stop
reorders close cars. The lap frame now carries the timing data's `pit_in_time_s` (in-lap row)
and `pit_out_time_s` (out-lap row), and `pit_loss.stop_table` each stop's `lane_s` and `team`.

- `replay: actual` (hindsight, ledger `pit_stops_replay`): a whole-race replay of a real
  strategy in the fidelity tool charges each real stop the pit loss plus its lane time less
  the race's median green lane time (`pit_loss.lane_deltas`). A stop's extra time is added
  after the simulation, so a stop just before a safety car keeps it (the SC queue, when on,
  would erase it).
- `team: true`: every stop a simulation prices is the pit loss plus the car's team's typical
  stop: the median lane-time difference of its green stops, shrunk toward 0 by n / (n + 3)
  (`pit_loss.team_delta`). The judge reads every stop of the race (ledger `pit_stops_team`,
  this race whole); a windowed fit (the belief) the stops completed when the car completed its
  calibration lap (`pit_loss.snapshot_time`, the same cut as the as-of pit loss), so nothing
  after the snapshot: `tests/test_sim_pit_stops.py` alters every other car's rows after the
  cut and the belief's times stay identical (the judge's change: the canary).

Measured (19 calibration races, main 2c7665b with this branch, the adopted judge and belief;
A/B sections without the hindsight variants, leave-one-stint-out or components; race-cluster
95% intervals; base close-pair order 0.753, Spearman 0.958, relative gap 13.17 s, abs 18.9 s,
B2 rho 0.866, regret 4.12 s, pit-now 0.810):

| arm | close-pair order | Spearman | rel gap s | abs MAE s | B2 rho | B2 regret s | pit-now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replay actual | +0.016 [-0.007, +0.043] | -0.001 | +0.02 [-0.31, +0.36] | -0.64 | 0 | 0 | 0 |
| team typical (judge) | +0.000 [-0.013, +0.014] | +0.001 | -0.01 | -0.21 | +0.000 | +0.004 | +0.000 |
| both, belief's as-of team stop too | +0.016 | -0.001 | +0.02 | -0.64 | +0.000 [-0.034, +0.031] | -0.001 | +0.001 |

Decision. `replay: actual` moves the close-pair order the way the noise floor said real stops would
(+0.016; its interval includes 0) and leaves the gaps alone. It touches only the replays of real
strategies, so the eval's scoring of hypothetical plans is unchanged: it can be adopted as a
measurement input (ledger `pit_stops_replay`, this race whole). `team: true` changes nothing
measurable (team medians differ by tenths after shrinkage, the same for every plan of a car
except through the stop count): not adopted. Both stay off by default.

The noise floor on this main (19 races, 368 close pairs, the adopted judge): judge 0.753
[0.698, 0.793] / 13.2 s; + per-stint pace 0.696 / 11.9 s; + pit laps 0.739 / 11.4 s; + SC/VSC
laps 0.614 / 5.2 s; + lap 1 (the ceiling) 0.745 [0.682, 0.800] / 3.2 s [2.6, 3.9]; pit laps alone
0.755 / 12.6 s. The judge's order is at the ceiling; the room is in the gaps, almost all of it on
the neutralised laps.

Two things the floor's levels take as given. Level 4 hands the replay the recorded lap 1, which
holds every place won or lost at the start: start-lap position changes are outside the model.
And every level prices each stint from its recorded starting tyre age: replays (`race_stints`,
the stint's first TyreLife less one) and snapshots (`snapshot_at` / `snapshot_as_of`, the
current set's TyreLife) already carry used sets (36% of first stints start on a used set, mean
age 1 lap, qualifying sets about 3), so the starting tyre state needed no change; a
hypothetical future stint in a plan starts on a new set (start age 0) unless the plan says
otherwise.

### The safety-car queue

`src/sim/sc_queue.py`, switch `SimConfig.sc_queue` (`enabled`, `mode`, `spacing_s`), off by
default. Under a safety car the field closes up behind it: whatever the gaps were when it came
out, at the end of its last lap the cars run nose to tail (on the calibration races the median
consecutive gap at the line at the end of an SC period is 0.49 s and the whole field is within
7 s, lapped cars included because they are waved past; under a VSC the gaps are held, 4.8 s
median before and after). A replay that keeps each car's own time through the period keeps the
gaps the race erased, which is where most of the judge's relative gap error comes from (the
noise floor above). What the queue changes is GAPS and ORDER, so it acts on the cars' race times,
not on any car's lap times. At each SC period S..E (S >= 2; a lap counts when most of the
field's rows carry status 4):

- order: each car's race time at the end of lap S - 1 plus the pit loss of its stops in S..E (a
  stop under the safety car costs places in the queue, not open-air seconds);
- the first car's time at the end of lap E is its order time plus the period's length (the judge:
  the recorded leader's time over S..E; a drawn period or a windowed fit: the car's own simulated
  neutralised laps);
- every other car lands `spacing_s` (0.5 s, the measured median above, set once) behind the car
  ahead in that order, or closer if it already was; a car that finished laps down (the recorded
  classification, judge only) stays out of the queue; racing resumes from there.

Two modes. `single` (the engine, what the judge and the belief use when scoring one car's
plans): `sc_queue.FieldQueue` places the simulated car among the OTHER cars' race. The judge
reads their recorded times at S - 1 and their stops under the period, the recorded period length,
and, when the engine carries no pace term (the eval's judge), the car's whole-race pace for its
key, since queue order is a fact about the car's real speed. The belief reads only the rows
visible at the snapshot (`Simulator.window_rows`): for a period in progress the others' times
at S - 1, and for a later one the field at the snapshot projected with its gaps to the car held,
plus the car's as-of pace. A plan that pits under a recorded or drawn safety car is then charged
the places it drops in the compressed field (spacing_s each), not the reduced open-air pit loss
alone; a plan that pitted just before one gets its lost time back up to the queue. `field` (the
fidelity tool's whole-race replays only): every replayed car is queued together (`sc_queue.apply`),
the version the whole-field replay (`src/sim/field_replay.py`) would use. In the fidelity tool a
single-mode replay runs with the car's pace in the engine (`with_pace`, a first pass's offset),
because the queue erases the pace gaps built up before each safety car and the pace term can no
longer be added over every lap afterwards. Ledger: `sc_queue` (judge: this race, whole; belief:
this race, to snapshot) and `sc_queue_spacing` (history). Future blindness:
`tests/test_sc_queue_future_blindness.py` alters every other car's rows after the cut and the
belief's times under an SC in progress and a later one stay identical (the judge's change: the
canary).

Measured (18 calibration races, main 606b6ab with the queue branch, A/B sections without the
hindsight variants, leave-one-stint-out or components; change against the judge with hybrid
SC/VSC, which the default adopted after this run; race-cluster 95% intervals):

| arm | close-pair order | Spearman | rel gap MAE s | abs MAE s | B2 rho | B2 regret s | pit-now agree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid judge (reference) | 0.729 | 0.960 | 13.2 | 19.6 | 0.835 | 4.60 | 0.778 |
| + queue (single) | 0.655 (-0.074) | 0.880 | 9.1 (-4.0) | 14.4 | 0.823 | 3.84 | 0.796 |
| + queue in the belief too | 0.655 | 0.880 | 9.1 | 14.4 | 0.845 | 4.79 | 0.702 |
| + traffic (earned passes), no queue | 0.699 | 0.955 | 13.1 | 19.5 | 0.827 | 4.89 | 0.774 |
| + queue + traffic | 0.619 | 0.856 | 9.6 | 14.2 | 0.821 | 4.14 | 0.777 |
| + queue + traffic, belief too | 0.619 | 0.856 | 9.6 | 14.2 | 0.835 | 5.40 | 0.682 |

Against the judge without SC/VSC (the base then): queue (single) order -0.060 [-0.104, -0.010],
gaps -5.3 s [-11.1, -0.6]; queue (field) order -0.027 [-0.061, 0.000], gaps -6.6 s [-12.6, -1.4].

Decision: NOT adopted (off by default). The queue improves relative gaps by about 4 to 6 s but
worsens close-pair order, and the earned-pass traffic does not recover it. Per race the order
loss sits in two races, 2022 Bahrain (Spearman 0.75 to -0.05) and 2022 Brazil (0.97 to 0.36):
cars close to a lap down when the safety car came out (90 s behind a 97 s lap) did not close up
to the queue in the race (the real field at the end of the Bahrain SC spans 38 s), while the
rule moves every car that finished on the lead lap into the train. Everywhere else gaps fall
(2022 Canada 31.7 to 4.5 s, 2019 Britain 13.4 to 4.1 s) with order level. The pit-now loss
when the belief runs the queue too is the two simulators' safety-car expectations, not the
queue: the belief draws later safety cars at its as-of rates and keeps an SC in progress going
for drawn laps, so a stop now (refunded by a later queue) or a stop under the SC (cheap in the
queue) looks better to it than to the judge, whose SC in progress ends when it really did
(2022 France at laps 20 to 40: the belief's pit-now plans 7 to 25 s ahead of staying, the
judge's 0.4 to 9 s behind). Before it can be adopted the queue needs the near-lapped cars'
rule (who joins the train, from the gap at deployment against the SC lap) and the two
simulators' SC duration and rate models aligned (the next section).

#### The near-lapped rule

Switch `sc_queue.held_behind_laps` (None, the default: as above; 1.25: the rule). What decides
whether a car joins the train, read from the 8 SC periods with S >= 2 of the calibration races
(`sc_queue.deployment_behind`: each car's distance behind the leader when the safety car came
out, from the line crossings made before the deployment, against its recorded gap at the end of
the period):

- Every car less than one lap behind at the deployment closed up (about 0.5 s per place),
  whatever the period length (the 3-lap periods of 2021 Spain and 2022 France included). The
  only cars that did not pitted on the last SC lap.
- Most cars just over a lap behind closed up too: they are waved past and reach the back of the
  train in time (2022 Bahrain 1.08 to 1.13 laps, 2022 Brazil up to 1.23, 2022 Canada up to 1.22).
- Cars 1.3 to 1.6 laps behind did not: 2022 Bahrain's STR, ALB, RIC, NOR, LAT and HUL (1.31 to
  1.56 laps) ran green laps under the SC on its last two laps and finished the period 24 to 38 s
  behind the leader; 2022 Brazil's LAT (1.55) 15 s. Only the longest period (2022 Canada, 7 laps)
  brought cars that far back (1.32, 1.77) into the train.
- 2021 Abu Dhabi is the exception: the cars 1.06 to 1.25 laps behind were let past only on the
  last lap and finished it 59 to 95 s back.
- Pitting under the safety car does not decide it: the stop is already priced in the queue
  order.

The old rule's `lapped` set holds only cars that finished laps down. The six Bahrain cars unlapped
later, so each was pulled to 0.5 s per place in a queue where the other cars kept their
recorded 24 to 38 s, and was placed ahead of the rest of its group.

The rule. A car more than `held_behind_laps` laps behind at the deployment does not join the train.
It keeps its gap, in the queue order, to the train's last car ahead of it. 1.25 is the cut that
separates the cars that closed up from those that did not on the recorded end-of-period gaps
(between Brazil's 1.23, which joined, and Bahrain's 1.31, which did not; 7 of 78 cars that did
not pit are misclassified, 5 of them 2021 Abu Dhabi's). It was fitted on the race data, not on a
scorecard result. Under the rule the single-car queue also:

- keys the other cars by their green time at S - 1, projected from their crossings before the
  deployment (`sc_queue.deployment_times`). A car's recorded crossing of S - 1 made under the
  safety car would have put it behind cars it was ahead of: 2022 Brazil's ALB and LAT were placed
  near the front of the train;
- leaves out of the field the cars that had retired. `FieldQueue.at` had projected them forward
  at `lap_ref_s` a lap, ahead of every car;
- ends the train where the recorded leader ended lap E, whatever lap the car's own count saw the
  safety car from. A car a lap down counts its laps beside the leader's next ones and had been
  given its own simulated SC laps as the period length.

The old rule is unchanged (`held_behind_laps` None, bit for bit). The judge reads the recorded
race. The belief reads only the rows visible at the snapshot: the deployment when they show it,
else the distances from the times at the end of lap S - 1, with the field projected with its gaps
held (`tests/test_sc_queue_future_blindness.py` covers the rule too).

Measured on 81a549f (19 calibration races, none failed; the adopted judge with hybrid SC/VSC as
the base; A/B sections without the hindsight variants, leave-one-stint-out or components;
race-cluster 95% intervals; `python -m src.sim.fidelity ablate --only sc_queue sc_queue_held
sc_queue_field sc_queue_held_field sc_queue_both sc_queue_held_both`; files in
`reports/calibration/fidelity/near-lapped-81a549f/`: the ablation, per-race measures, the
whole-field replay per race and the deployment distances):

| arm | close-pair order | Spearman | rel gap MAE s | B2 rho | B2 regret s | pit-now agree |
| --- | --- | --- | --- | --- | --- | --- |
| queue off (the adopted judge) | 0.753 | 0.958 | 13.17 | 0.868 | 4.11 | 0.816 |
| old queue (single) | 0.690 (-0.062 [-0.115, -0.008]) | 0.882 | 9.35 (-3.82 [-7.83, -0.87]) | 0.859 | 3.77 | 0.803 |
| near-lapped rule (single) | 0.709 (-0.043 [-0.081, -0.010]) | 0.953 | 7.47 (-5.70 [-10.91, -1.50]) | 0.860 | 3.95 | 0.807 |
| old queue (field) | 0.726 (-0.027 [-0.044, -0.010]) | 0.954 | 7.70 (-5.47) | | | |
| near-lapped rule (field) | 0.734 (-0.019 [-0.036, -0.003]) | 0.956 | 7.45 (-5.72 [-10.92, -1.47]) | | | |
| old queue, belief too | 0.690 | 0.882 | 9.35 | 0.880 | 4.64 | 0.715 |
| near-lapped rule, belief too | 0.709 | 0.953 | 7.47 | 0.877 | 4.80 | 0.723 |

Per race (single / field), close-pair order, Spearman, relative gap:

| race | queue off | old queue | near-lapped rule | rule, field |
| --- | --- | --- | --- | --- |
| 2022 Bahrain (46 pairs) | 0.804, 0.754, 39.8 s | 0.609, -0.054, 35.0 s | 0.630, 0.754, 8.9 s | 0.761, 0.768, 7.6 s |
| 2022 Brazil (48 pairs) | 0.854, 0.975, 28.2 s | 0.646, 0.368, 13.9 s | 0.792, 0.919, 3.2 s | 0.833, 0.963, 3.8 s |

Snapshots with a neutralisation in progress (lap k or k + 1 neutralised in the recorded race;
belief against judge):

| arm | SC: pit-now agree (60) | SC: B2 rho | SC: regret s | VSC: pit-now agree (48) |
| --- | --- | --- | --- | --- |
| queue off | 0.683 | 0.819 | 10.75 | 0.646 |
| near-lapped rule in the judge | 0.733 | 0.776 | 10.60 | 0.667 |
| near-lapped rule in both | 0.683 | 0.819 | 11.77 | 0.500 |
| old queue in both | 0.567 | 0.821 | 12.68 | 0.521 |

Whole-field replay (`src/sim/field_replay.py`, every classified car's real strategy from k, the
world as a file with the switch, leave-one-driver-out fit, traffic off; the single-car queue
against the other cars' recorded race; change against queue off, race-cluster intervals):

| pace, k | close pairs off | old queue | near-lapped rule | rel gap off s | old queue | near-lapped rule |
| --- | --- | --- | --- | --- | --- | --- |
| asof, 0 | 0.538 | -0.008 [-0.046, +0.028] | +0.003 [-0.028, +0.038] | 27.1 | -1.84 [-4.53, -0.23] | -2.21 [-4.91, -0.32] |
| asof, 10 | 0.655 | -0.014 [-0.077, +0.044] | -0.014 [-0.048, +0.020] | 22.3 | -5.34 [-10.76, -1.19] | -5.44 [-10.41, -1.50] |
| asof, 20 | 0.668 | -0.014 [-0.032, +0.000] | -0.014 [-0.053, +0.016] | 19.9 | -5.21 [-10.84, -1.05] | -5.59 [-10.71, -1.34] |
| asof, 30 | 0.720 | -0.035 [-0.093, +0.014] | -0.033 [-0.075, +0.011] | 16.2 | -3.73 [-8.98, +0.06] | -4.13 [-9.29, +0.00] |
| asof, 40 | 0.753 | -0.038 [-0.095, +0.004] | -0.035 [-0.071, +0.000] | 13.6 | -3.90 [-8.91, -0.07] | -4.25 [-9.16, -0.06] |
| practice, 10 | 0.582 | -0.046 [-0.089, -0.003] | -0.049 [-0.084, -0.009] | 23.1 | -1.35 [-4.34, +0.88] | -1.88 [-5.24, +0.72] |
| practice, 20 | 0.609 | -0.060 [-0.114, +0.000] | -0.054 [-0.107, +0.004] | 21.1 | -2.00 [-4.32, +0.00] | -2.72 [-5.97, -0.02] |
| practice, 40 | 0.701 | -0.073 [-0.133, +0.000] | -0.071 [-0.140, -0.006] | 15.4 | -2.59 [-5.99, +0.18] | -3.11 [-6.90, +0.00] |

Spearman on the whole-field replay, 2022 Brazil at practice pace: off 0.86 to 0.92, old queue
0.30 to 0.42, the rule 0.85 to 0.90 (relative gap 25 s to 4 to 13 s). 2022 Bahrain: off 0.62 to
0.75, old queue 0.36 to 0.57, the rule 0.42 to 0.61 (relative gap 18 to 37 s to 12 to 15 s).

Reading under the selection rule:

- The rule removes the near-lapped failure. Spearman returns to the queue-off level (0.953
  against 0.958; the old queue 0.882), Brazil's order is back (0.919 single, 0.963 field) and the
  gap gain grows: relative gap 13.2 to 7.5 s, against 9.4 s for the old queue. The noise floor
  with neutralised laps recorded is 5.2 s.
- Close-pair order is still below queue-off: -0.043 [-0.081, -0.010] single, -0.019 [-0.036,
  -0.003] field. It is NOT an adopt-candidate: the rule requires order at least level with
  queue-off within the interval.
- What is left of the single-car loss is how a simulated car is placed among the other cars'
  recorded race, not who joins the train. A car whose simulated time at S - 1 has drifted from
  its real one loses or gains places in the train for it, and keeps them. 2022 Bahrain's leader
  LEC had drifted 46 s by lap 45 on the whole-field replay (k = 20, practice pace) and pitted
  under the SC, so it rejoined 13th. Off, the same drift stays a time error that the relative
  measure partly removes. 16 of the 368 close pairs are lost: 8 in Bahrain, 3 in Brazil, 2 in
  2019 Britain, 1 each elsewhere. The field mode queues simulated cars among simulated cars and
  loses 7.
- The belief with the rule: pit-now agreement on SC-in-progress snapshots is level with queue-off
  (0.683; the old queue in both 0.567). Overall it is -0.093 [-0.143, -0.043], the same cause as
  above (the two simulators' safety-car expectations), and on VSC snapshots 0.500 against 0.646.
- 2021 Abu Dhabi's relative gap gets worse under every queue (22.8 to 27.2 s): its cars 1.06 to
  1.15 laps behind were held by race control, and no rule fitted on the other races reproduces
  that.

#### Placing the car by the record

Switch `sc_queue.placement` (`clock`, the default; `record`). It applies with `held_behind_laps`
set and in the judge only (fit_window full; hindsight, ledgered as `sc_queue_placement`).

The near-lapped rule's remaining order loss came from placing the simulated car in the train by
its simulated time at S - 1 among the other cars' recorded times. Any drift of its simulated clock
by then turned into places in the train, and the car kept them. Under `record`:

- the car is placed by its recorded green time at S - 1, from its own crossings before the
  deployment (`sc_queue.deployment_times`);
- each plan is moved by its simulated time difference at S - 1 to the plans of the same batch
  that stopped where the car really did before S (the median of those plans' times). A replay of
  the real strategy lands at its recorded place; a plan that stopped earlier or later moves by
  what the simulator says that cost or gained it;
- with no such plan in the batch, the car keeps its clock;
- for a period already running at the snapshot, the car's recorded place is used.

Measured on a3e2a46 (main 3ed475f, the v3 defaults: belief cliff edge at its point, the judge's
scaled wear guard), the same 19 races and options as above. The base is close pairs 0.753,
Spearman 0.957, relative gap 13.19 s, B2 rho 0.882, regret 3.36 s, pit-now 0.834.

| arm | close-pair order | Spearman | rel gap MAE s | B2 rho | B2 regret s | pit-now |
| --- | --- | --- | --- | --- | --- | --- |
| near-lapped rule (clock, single) | 0.709 (-0.043 [-0.081, -0.010]) | 0.953 | 7.49 (-5.70) | -0.010 | -0.20 | -0.009 |
| near-lapped rule (field) | 0.734 (-0.019 [-0.036, -0.003]) | 0.956 | 7.47 (-5.72) | 0 | 0 | 0 |
| rule + placement by the record | 0.747 (-0.005 [-0.033, +0.021]) | 0.960 | 7.60 (-5.59 [-10.72, -1.47]) | -0.010 [-0.015, -0.006] | -0.20 [-0.38, -0.03] | -0.008 [-0.022, +0.006] |
| the same, the rule in the belief too | 0.747 | 0.960 | 7.60 | +0.008 | +0.55 | -0.116 [-0.175, -0.063] |

Per race, queue off / the rule (clock) / the rule with placement:

- 2022 Bahrain: close pairs 0.804 / 0.630 / 0.761; Spearman 0.754 / 0.754 / 0.821; relative
  gap 39.8 / 8.9 / 9.6 s.
- 2022 Brazil: close pairs 0.854 / 0.792 / 0.875; Spearman 0.975 / 0.919 / 0.973; relative gap
  28.2 / 3.2 / 3.6 s.
- 2022 Canada: close pairs 0.733 / 0.700 / 0.800.
- 2021 Abu Dhabi: close pairs 0.786 / 0.821 / 0.893.

Pit-now agreement on SC-in-progress snapshots (60): off 0.700, placement in the judge 0.750,
placement in the judge with the rule in the belief 0.683. On VSC snapshots (48): 0.646, 0.667,
0.500.

Whole-field replay against queue off. Close pairs:

| pace, k | the rule (clock) | the rule with placement |
| --- | --- | --- |
| asof, 0 | +0.003 | +0.049 [+0.000, +0.102] |
| asof, 10-40 | -0.014 to -0.035 | +0.005 to +0.030 (intervals include 0) |
| practice, 0-40 | -0.049 to -0.071 at k 10-40 | -0.033 to +0.049 (intervals include 0) |

Relative gap with placement: -2.5 to -6.0 s, intervals clear of 0 at most k.

dev20 rescore, the baselines' regret against each judge's own reference (judge variants
`configs/sim/world_hybrid.yaml` plus the switch, belief default, references and baselines rebuilt
at a3e2a46; paired by scenario, race-cluster 95%). The queue-off judge rebuilt here is the
current default, not the committed dev20 files.

| policy | off s | rule (clock) | rule with placement |
| --- | --- | --- | --- |
| belief_optimizer | 3.25 | -0.14 [-0.29, +0.01] | -0.21 [-0.38, -0.05] |
| run_to_flag | 8.80 | +0.34 [+0.17, +0.56] | +0.26 [+0.05, +0.49] |
| one_stop_total | 9.64 | +0.26 [+0.07, +0.49] | +0.18 [-0.04, +0.43] |
| actual_strategy | 6.58 | -0.12 [-0.31, +0.07] | -0.15 [-0.32, +0.03] |

The judge's mean reference stop count rises from 0.789 to 0.838: a stop under a safety car now
costs places in the train, not open-air seconds, so the reference takes more stops under SC.

Reading under the selection rule:

- Placement by the record is an adopt-candidate for the judge. Close-pair order is level with
  queue-off within its interval (-0.005 [-0.033, +0.021]), Spearman is level, and the gap gain
  holds: relative gap 13.2 to 7.6 s.
- The belief's plan ranking against the judge moves by -0.010 in rho and -0.20 s in regret
  (better), and pit-now agreement is unchanged within its interval.
- On dev20 the belief optimizer's regret falls by 0.21 s; the fixed-rule baselines' regret rises
  by 0.2 to 0.3 s.
- The queue stays off in the belief: there it costs 0.116 of pit-now agreement (the two
  simulators' safety-car expectations).
- To adopt: add to the judge's config (`configs/sim/world_hybrid.yaml`, `score.default_world_config`):

  ```yaml
  sc_queue: {enabled: true, mode: single, held_behind_laps: 1.25, placement: record}
  ```

  This changes the judge, so every reference and result scored on it is stale.

Files: `reports/calibration/fidelity/placement-a3e2a46/` (the ablation, per-race measures, the whole-field replay per race, and the dev20 regret per scenario and policy under the three judges).

### Safety-car duration and rates: the belief against the judge

Switches `events.params.duration` ("geometric", the default, or "history") and, for the judge's
later periods, `pooled` (events.RateEvents; race_history.period_lengths, pooled_event_rates).

Measured first (19 calibration races, the periods of the races' own track status):

| | predicted | real |
| --- | --- | --- |
| SC still to run after 1, 2, 3, 4, 5, 6 laps (belief, memoryless geometric, as-of mean 3.3 to 4.7) | 3.9, 3.9, 3.9, 4.0, 4.0, 4.0 laps | 3.7, 2.7, 1.7, 1.7, 1.0, 0.2 laps |
| SC rest: bias / 90% coverage (belief, geometric) | +1.84 laps / 0.79 | |
| VSC rest: bias / 90% coverage (belief, geometric) | +1.66 laps / 0.55 | |
| SC rest, history model as of the race (L given L >= laps so far) | bias -0.35 laps / coverage 0.92 | |
| VSC rest, history model as of the race | bias +0.19 laps / coverage 1.00 | |
| later SCs over the 19 races (belief as-of rates / judge's EventModel defaults / pooled history) | 16.2 / 16.2 / 13.0 | 11 |
| later VSCs (same) | 16.2 / 21.6 / 14.5 | 15 |

The belief's rates are within about 1.3 Poisson sd of the record (not changed); its period
lengths were biased: a memoryless geometric keeps every safety car in progress going for about
four more laps whatever its age, so the belief priced a stop under it as cheap for longer than the
race allowed. The judge's later periods used EventModel's provisional 2021-Spain defaults (VSC
rate 44% high). "history" draws a period's length from the dry races' recorded lengths (as of
the race for the belief: race_history.earlier_races, the provenance registry's `event_rates`
artifact; pooled over the table for the judge), shrunk toward an a priori geometric with
EVENT_PRIOR's mean worth two periods (the documented prior for a race with no earlier one); a
period in progress for e laps continues by L - e laps with L drawn given L >= e (0: it ends at
the snapshot). Common random numbers are kept: the draws use the scenario's event substream.

Scorecard (19 races, main a94177d with this branch, A/B sections without the hindsight
variants, leave-one-stint-out or components; change against the adopted base):

| arm | judge order / gaps | B2 rho | B2 regret s | pit-now | B1 MAE s | B3 cov 90 |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.753 / 13.17 s | 0.866 | 4.122 | 0.810 | 50.1 | 0.976 |
| belief duration history | unchanged | +0.001 [-0.000, +0.002] | +0.007 [-0.081, +0.094] | +0.003 [-0.012, +0.019] | 41.8 | 0.991 |
| judge later periods pooled | unchanged | +0.001 [-0.002, +0.003] | -0.015 [-0.060, +0.023] | +0.004 [-0.002, +0.010] | 50.6 | 0.975 |
| both (aligned) | unchanged | +0.002 [-0.000, +0.004] | -0.011 [-0.107, +0.085] | +0.006 [-0.008, +0.020] | 42.9 | 0.991 |

Split by the track at the snapshot (panel snapshots where both a pit-now and a stay plan are
legal; "value" = best pit-now plan less best other plan, belief less judge):

| snapshots | n | value gap base -> aligned s | pit-now agree | regret s | rho |
| --- | --- | --- | --- | --- | --- |
| green | 1083 | +0.01 -> -0.47 | 0.815 -> 0.821 | 3.78 -> 3.80 | 0.868 -> 0.870 |
| SC or VSC in progress | 89 | -6.04 -> -3.61 | 0.742 -> 0.753 | 9.25 -> 9.26 | 0.839 -> 0.840 |
| of which SC | 47 | -6.13 -> -5.71 | 0.872 -> 0.745 | 13.88 -> 13.51 | 0.800 -> 0.802 |
| of which VSC | 42 | -5.95 -> -1.27 | 0.595 -> 0.762 | 4.07 -> 4.51 | 0.884 -> 0.883 |

Reading: aligned, every overall measure improves or holds and the judge's order and gaps do not
move (whole-race replays use the recorded periods). Under a VSC the belief's overvaluation of a
stop falls from 6 s to 1.3 s and pit-now agreement rises from 0.60 to 0.76. Under an SC the
value gap shrinks only a little (6.1 to 5.7 s) and pit-now agreement falls by six of 47
snapshots: four are 2019 Britain, the earliest race, which has no earlier history and runs on the
a priori prior (an SC may end on any lap, so a stop now looks less certain to be cheap), two are
2022 Brazil, where the value is within 2 s of zero either way. What is left under an SC is the
judge knowing the real end of the period in progress (it often ends at the next lap, so a stop
now is a green stop): no as-of model can know that, and the remaining gap is that hindsight.
Adopted 2026-11-22 as the default on both sides (the current configuration below; the earlier
pair is `configs/sim/legacy/world_pre_sc_alignment.yaml` and `belief_pre_sc_alignment.yaml`, the
ablation toggle `sc_alignment_off`). Limitation, stated and not chased: under an SC in progress
pit-now agreement is lower than with the geometric (six of 47 snapshots), mostly a race with no
earlier history (2019 Britain, on the a priori prior) and values within 2 s of zero; the value
gap that remains under an SC is the judge's knowledge of when the real period ends.

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

### The current configuration (adopted from the scorecard)

The defaults everywhere the evaluation reads one (`score.race_simulators`, `src/eval/rerun.py`
`default`, the agent's tools, the scorecard's base):

- **Judge (world):** linear_fuel with driver effects, no hinge (`config.WORLD_DEGRADATION`); the
  wear cliff with the tyre-life edge from history only (`config.WORLD_EDGE` `history_only`); hybrid
  SC/VSC (`score.WORLD_EVENTS`, `WORLD_EVENTS_DEFAULT` True), drawn on the events substream so a
  policy's plan and the reference see the same periods for a seed, the later periods at rates and
  lengths pooled over the history table (`duration: history`, `pooled: true`, adopted
  2026-11-22); the thin-fit wear guard with the fit's sd scaled by the number of drivers
  (`config.WORLD_WEAR_GUARD`, `wear_guard: {enabled: true, scale_sd_by_drivers: true}`, adopted
  2026-11-22, "Thin wear fits"); the safety-car queue (`config.WORLD_SC_QUEUE`, `sc_queue: {enabled:
  true, mode: single, held_behind_laps: 1.25, placement: record}`, adopted 2026-11-22, "Placing the
  car by the record"): at each recorded SC period the simulated car joins the train at its recorded
  place, cars more than 1.25 laps behind at the deployment held at their gap (hindsight, the judge
  only, ledgered as `sc_queue` and `sc_queue_placement`); the tyre sets at the snapshot
  (`config.WORLD_TYRE_INVENTORY`, `tyre_inventory: true`, adopted 2026-11-22, "Tyre sets at the
  snapshot"; the same as-of inventory as the belief, ledgered as `tyre_inventory`); no traffic; R8
  retired. As a file: `configs/sim/world_hybrid.yaml`. config_hash `c821efe0bc98` (before the tyre
  sets `7c974c32c67c`, before the SC queue `15e3f31d98a7`, before v3 `5069228b35e7`, before the SC
  alignment `abc455a9bbb6`, before the scorecard `d230d3a56b0b`).
- **Belief:** `configs/sim/belief.yaml`: linear_fuel with driver effects, wear bound 0, fuel
  anchoring, persistent lap noise, as-of pit loss, rates and tyre-life edge (history extended by the
  ages seen at the snapshot), the as_of_history calibrator, no traffic, `plan_on` draws, no
  `sd_scale`, SC/VSC lengths from the as-of history of period lengths, a period in progress
  continuing given its laps so far (`events.params.duration: history`, adopted 2026-11-22), the
  wear-cliff edge at its point in every sample (`belief_cliff_edge: point`, adopted 2026-11-22,
  "Belief priors"); the tyre sets at the snapshot (`tyre_inventory: true`, adopted 2026-11-22, "Tyre
  sets at the snapshot"); no safety-car queue (`tests/test_eval_default_config.py` checks no belief
  config carries it). config_hash `aaf95a09bc50` (before the tyre sets `9d691bbbcf24`, before v3
  `351f4124ff22`, before the SC alignment `f7b872d14c0c`, before the scorecard `d479652b0fe7`). `belief_traffic.yaml` and
  `belief_learned.yaml` are it plus one block.
- **Kept off, as named switches:** traffic on either side (`configs/sim/world.yaml`,
  `belief_traffic.yaml`), the learned calibrator (`belief_learned.yaml`), the hinge
  (`world_de_hinge.yaml`), `sd_scale`, `plan_on: point`, the safety-car queue in the belief, the
  pit-stop durations (`pit_stops`), overtaking difficulty, three-stop search.
  The agent protocol stays `v2free`. The earlier pairs: `configs/sim/legacy/world_pre_scorecard.yaml`
  and `belief_pre_scorecard.yaml`; `world_pre_sc_alignment.yaml` and `belief_pre_sc_alignment.yaml`;
  `world_pre_v3.yaml` and `belief_pre_v3.yaml`; `world_pre_sc_queue.yaml` (the belief unchanged);
  `world_pre_tyre_inventory.yaml` and `belief_pre_tyre_inventory.yaml`.
  Ablation toggles back off: `wear_guard_off`, `wear_guard_unscaled`, `belief_cliff_edge_draw`,
  `sc_queue_off` (and `sc_queue_clock`, `sc_queue_old`, `sc_queue_field`, `sc_queue_both` for the
  queue's other forms), `tyre_inventory_off` (both simulators).

Confirmation after the tyre sets on 3881500 (`reports/calibration/fidelity/tyre-inventory-3881500/`, all 19
calibration races, none failed). Hashes and versions:

| | before | after |
|---|---|---|
| SIM_CODE_VERSION | `32671b6a2ad5` | `30efab2adf7a` |
| default world config_hash | `7c974c32c67c` | `c821efe0bc98` |
| belief config_hash | `9d691bbbcf24` | `aaf95a09bc50` |

Measures; the judge's are unchanged because its whole-race replays already run real ages:

| measure | after | before |
|---|---|---|
| close pairs out of sample | 0.747 | 0.747 |
| Spearman | 0.960 | 0.960 |
| relative gap | 7.60 s | 7.60 s |
| plan-rank agreement | 0.880 | 0.872 |
| judge-regret of the belief's favourite | 3.47 s | 3.16 s |
| pit-now agreement | 0.847 | 0.827 |
| belief replay close pairs | 0.667 | unchanged |
| plan 90% coverage | 0.972 | |
| E_real_regret_s | 5.21 s | 5.59 s |

The noise floor (`ceiling_pooled.csv` there) is unchanged. The paired off/on table and the reading are
under "Tyre sets at the snapshot". Every reference built on the earlier world is stale: its world
config_hash and SIM_CODE_VERSION both changed.

Confirmation after the judge's safety-car queue on 83fb72e
(`reports/calibration/fidelity/sc-queue-83fb72e/`, all 19 calibration races, none failed, future
blindness pass; SIM_CODE_VERSION `32671b6a2ad5`, before `3c48388e8271`; default world config_hash
`7c974c32c67c`, before `15e3f31d98a7`; the belief unchanged, `9d691bbbcf24`):

- close pairs out of sample 0.747 (before 0.753);
- Spearman 0.960 (0.957);
- relative gap 7.60 s (13.19);
- absolute 13.4 s (18.9);
- plan-rank agreement 0.872 (0.882);
- judge-regret of the belief's favourite 3.16 s (3.36);
- pit-now agreement 0.827 (0.835);
- belief replay close pairs 0.667 (unchanged);
- plan 90 % coverage 0.975.

Noise floor (`ceiling_pooled.csv` there): judge 0.747 [0.683, 0.798] / 7.6 s [5.7, 10.2],
+ per-stint pace 0.658 / 6.4 s, + pit laps 0.652 / 6.9 s, + SC/VSC laps 0.611 / 5.2 s, + lap 1
(the ceiling) 0.745 [0.682, 0.800] / 3.2 s [2.6, 3.9]. The judge's relative gap now sits 2.4 s
above the level with every neutralised lap recorded, down from 8.0 s. Every reference built on the
earlier world is stale (its world config_hash and SIM_CODE_VERSION both changed).

Confirmation after v3 (the wear guard and the point cliff edge) on 02a3a7f
(`reports/calibration/fidelity/v3-02a3a7f/`, all 19 calibration races, none failed;
SIM_CODE_VERSION `df1ab3800c69`, before `afd80306ad68`): close pairs out of sample 0.753, Spearman
0.957, relative gap 13.19 s, absolute 18.9 s, plan-rank agreement 0.882, judge-regret of the
belief's favourite 3.36 s, pit-now agreement 0.835, belief replay close pairs 0.667, B1 MAE 43.7 s,
plan 90 % coverage 0.988. Noise floor (`ceiling_pooled.csv` there): judge 0.753 / 13.2 s,
+ per-stint pace 0.696 / 11.9 s, + pit laps 0.739 / 11.4 s, + SC/VSC laps 0.614 / 5.2 s, + lap 1
(the ceiling) 0.745 [0.682, 0.800] / 3.2 s [2.6, 3.9]. Every reference built on the earlier world
is stale (its world config_hash and SIM_CODE_VERSION both changed).

Confirmation after the SC alignment on 997d69d (`reports/calibration/fidelity/sc-align-997d69d/`,
all 19 calibration races, none failed; SIM_CODE_VERSION `938d268003fa`): close pairs out of sample
0.753, Spearman 0.958, relative gap 13.17 s, absolute 18.9 s, plan-rank agreement 0.868,
judge-regret of the belief's favourite 4.11 s, pit-now agreement 0.816, belief replay close pairs
0.664, B1 MAE 42.9 s, plan 90 % coverage 0.991. Noise floor (`ceiling_pooled.csv` there): judge
0.753 / 13.2 s, + per-stint pace 0.696 / 11.9 s, + pit laps 0.739 / 11.4 s, + SC/VSC laps
0.614 / 5.2 s, + lap 1 (the ceiling) 0.745 [0.682, 0.800] / 3.2 s [2.6, 3.9].

SIM_CODE_VERSION `096753d89c0f` (before `ba80a54af80b`): `config.py` is a physics module. Every
reference built on the earlier world is stale (its key holds the world's config_hash and
SIM_CODE_VERSION); scenario sets under the as_of rule do not depend on the world.

Confirmation on c67c9a7 (`reports/calibration/fidelity/adopt-c67c9a7/`, the 19 calibration races
named; 2022_11_austria fails in every configuration, the base included, on a set with no recorded
compound at OCO's lap 45, so 18 races count). Headline: close pairs out of sample 0.751, Spearman
0.957, relative gap 13.6 s, plan-rank agreement 0.867, judge-regret of the belief's favourite
4.2 s, pit-now agreement 0.806, belief replay close pairs 0.665, plan-gap coverage 68 / 90 %
0.743 / 0.865, plan 90 % coverage 0.975. Noise floor: level 4 (stints, stops, neutralised laps
and lap 1 as recorded) 0.745 close pairs and 3.2 s; the judge already sits at that order.

Each adopted choice switched back off (change against the new base, race-cluster 95% intervals):

| toggle | close pairs | Spearman | rel gap s | rank rho | regret s | pit-now | belief replay close pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| world_events_off (hybrid off) | −0.025 [−0.050, −0.007] | −0.009 | +2.41 [+0.44, +4.51] | −0.001 | +0.40 [+0.12, +0.70] | +0.002 | 0 |
| cliff_edge_history (edge extended by this race) | −0.022 [−0.068, +0.006] | +0.003 | −0.40 [−1.02, +0.07] | −0.003 | −0.44 [−1.05, +0.05] | −0.029 [−0.051, −0.007] | 0 |
| belief_no_driver_effects | 0 | 0 | 0 | −0.030 [−0.051, −0.008] | +1.05 [+0.31, +1.82] | −0.021 | −0.031 [−0.050, −0.016] |
| world_linear (judge: neither) | −0.014 [−0.036, +0.002] | −0.002 | −0.21 | −0.030 [−0.074, +0.005] | +0.81 [−0.53, +1.94] | +0.011 | 0 |
| world_hinge (judge: + hinge) | −0.033 [−0.065, −0.009] | −0.005 | −0.00 | −0.002 | +0.17 [+0.01, +0.33] | −0.006 | 0 |
| traffic_both (still rejected) | −0.041 [−0.084, +0.003] | −0.004 | −0.22 | −0.008 [−0.014, −0.003] | +0.07 | +0.006 | −0.002 |

Reading under the selection rule: hybrid SC/VSC, belief driver effects and no hinge each earn their
place with intervals clear of zero. Driver effects in the judge lead on order and ranking with
intervals that touch zero. The history-only edge leads on close pairs and pit-now agreement, while
the earlier edge is slightly better on relative gaps and regret; every one of those differences
except pit-now is inside its interval, so it stays adopted (no hindsight). Traffic stays off.

## Tests

The suite runs in three tiers (fast by default, `slow` for real races and end-to-end runs,
`legacy` for legacy switches; README.md "Testing"). `tests/test_eval.py` runs offline: synthetic races for sampling and scoring, plus the
committed Spain cache for the v1 set and an end-to-end baseline run.
`tests/test_eval_integration.py` covers the belief and world split and the decision unit,
`tests/test_eval_part2.py` positions, the drawn-events warning, the disagreement flag, the
eval split, `run-all` with two spawned workers, the agent policy, the snapshot contract and
the minimum-stint rule. `tests/test_eval_reference.py` covers the reference cache (key
stability, cached against computed rows, `--check` across code versions, hindsight
events, the d3 exception), `regret_se_s` on a world with known per-draw differences,
and `within_mc` at N = 1 against the tie rule.
`tests/test_optimizer_bridge.py` covers the limit-aware baselines (fixed_one_stop's moved
stop, greedy_crossover's forced stop, and every
baseline unchanged with R8 off). `tests/test_eval_rule_baselines.py` covers the rule
baselines on synthetic races and on 2021 Spain: the default `baselines` group unchanged, the
already-stopped and not-yet-stopped branches, a forced stop for R3 and for R8 (one stop and two),
the invalid row when no legal plan exists within `max_stops`, legality, determinism and regret
>= 0. `tests/test_sim_proxy.py` covers the compound proxy (which
sources enter the blend and with what weight, the weighted mean, anchoring, the qualifying gap
and its place in the model report, the ordering constraint and the wear floor, the proxied
covariance and draws, the belief's information boundary, the off
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

**Every run in this section is superseded** ("Result files"): pilot1, d2, Case B, the dev6 runs,
d4 at tag `d4-final-eval` and the rule baselines scored after it. They ran under the earlier
protocol, before the information-contract fixes. They are kept as the record of
how the agent and the protocol got here. The current development result is "Development result
(dev20) at eval-v2-dev2"; the held-out result is the rerun pipeline's `test9`.

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
run 2, made only to measure run-to-run variation; run 2 was the cited development result of that
setting (superseded, "Result files").
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
result (tag `d4-final-eval`) was produced by the loop before this fix. It was not rerun: it is
superseded, and the held-out result is the rerun pipeline's `test9` ("Result files").

### d4, the earlier held-out result (superseded)

**Superseded.** This run passed race-wide tyre life (R8) and other future information to the
policies ("Information contract"), so its numbers are not a result of the corrected protocol;
the held-out result is the rerun pipeline's `test9`, and the current development result is
"Development result (dev20) at eval-v2-dev2". Kept as the record of that run.

d4 was the report set and this was the held-out result, declared in advance and run once at tag
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

Superseded with the d4 run they follow (above). The two rules, in their current form (the as-of
tyre-life estimate in place of R8, "Rule baselines: run_to_flag and one_stop_total"), are the
lead comparisons of the dev20 result and of the held-out run.

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
