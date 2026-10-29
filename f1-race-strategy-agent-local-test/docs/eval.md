# Evaluation harness

`src/eval` scores any strategy policy against the optimizer from the same snapshot, over a
fixed, versioned set of decision points. No model is needed except for the `agent` policy.

```bash
python -m src.eval scenarios --races 2021_04_spain --version v1 --out data/scenarios/v1.json
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
| belief | `configs/sim/belief.yaml` | `to_snapshot`: race laps up to the snapshot, shrunk toward the practice prior; traffic off | every policy: the baselines plan on it through the bridge (`from_contract(snapshot, belief)`, `pace_for_crossover(belief, snapshot)`), and agent decisions are expanded on it |
| world | the profile's SimConfig | `full`: every race lap (hindsight); traffic off | the reference `optimize_batch` and every scoring call |

A policy is called as `policy(scenario, belief, max_stops)` and never sees the world. The
regret definition is unchanged: the policy's plan and the world optimizer's plan, both
scored by the world. With the default world, v1 stays comparable with the numbers before
the split. `configs/sim/world.yaml` is the world with ghost traffic on, for position regret
(below); pass it with `--world-config`. Ghost traffic replays the recorded field, so it
belongs to the world only, and only under the recorded events: an event model other than
`none` (or a fixed scenario other than the record) adds `traffic_recorded_with_drawn_events`
to every plan's support warnings (docs/traffic.md, limit 5).

Rows carry `belief_config_hash`, `world_config_hash` and `belief_window`. The config hash
leaves `fit_window` out, so the two hashes are equal under the defaults and
`belief_window` says which window the policies planned on.

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
  two-stop reference;
- `Simulator.expand_action` joins the two. A stay of n at least the laps left runs the
  current set to the flag.

A submission that cannot become a legal plan (no compound, a compound the belief does not
model, n below 1, no legal continuation) is an invalid row.

## The agent policy

`--policies agent` is `eval.policies.AgentPolicy`, the snapshot loop of `src/agent.py`, on
any scenario of any race (the lap-0 same-race guard is gone):

- the agent's view is `RaceSnapshot.from_contract(scenario.snapshot)` (`src/snapshot.py`):
  lap, tyre, gaps, `field.nearby`, `recent_lap_times_s`, weather, `race.min_stint_laps`
  and the inventory-legal compounds; the stint history and the sets stay out of it;
- `run_agent(view, verbose=False, max_steps=profile.agent.max_steps, log_path=...)` logs
  the trajectory to `reports/eval/trajectories/<set version>/<scenario_id>.jsonl` (":" in
  the id written "_"; a rerun replaces the file);
- the submission is scored as a decision (above); `tool_calls` is the trajectory length,
  `input_tokens` and `output_tokens` the backend's reported usage (blank when it reports
  none); a backend exception is an `error` row, not a crash.

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

### Development and report sets

`data/eval_set.json` is the registered eval split, written by `register_eval_set` (Sarah's
item): the races held out for reporting, `{"races": [...], ...}` or a bare list of ids.
While it exists, `build_set` (and `scenarios`, `--eval-set` for another file) refuses any
race outside it, records the file's sha256 as `sampling["eval_set_digest"]`, and marks the
set `development: false`. Without it a set is `development: true`.

- **v1 is the development set**: sampled before the split, from the race we tuned on.
  Its numbers are for debugging the harness and the baselines, not for the report.
- **v2, sampled from the eval races, is the report set**: the numbers the report quotes.

`scenarios --development` (`build_set(..., development=True)`) still samples a development
set once the split exists, a Spain debug set say: no race is refused, the set is marked
`development: true`, and `sampling["eval_set_digest"]` is recorded anyway so the set shows
the split existed. Report numbers come only from sets with `development: false`; `score`
prints one line naming which kind of set it scored.

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
| `latency_s`, `tool_calls`, `input_tokens`, `output_tokens` | per decision, when the policy reports them |
| `pit_now_opt` | the world optimizer's first stop is at the end of this lap (`opt_first_pit_lap == laps_completed + 1`) |
| `decision`, `p_pit` | agent rows only: PIT or STAY_OUT, and the stated probability that pitting now is right (`confidence` for PIT, `1 - confidence` for STAY_OUT) |
| `brier` | summary, agent rows: mean of `(p_pit - pit_now_opt)^2`. Only in tables of rows that have `p_pit`. |
| `pit_now_agreement` | summary, agent rows: share of decisions where PIT matches `pit_now_opt` |
| `final_position_mean`, `opt_final_position_mean` | world traffic on: the plan's and the optimizer's mean finishing position over the samples, 1 + the ghosts that finished the last lap before the plan's world time (`traffic.place`); a DNF sample is behind every finisher, ordered by `dnf_lap` against the ghosts that retired |
| `position_regret` | `final_position_mean - opt_final_position_mean`; positive means finishing behind the reference. NaN with world traffic off (the default world) or when traffic cannot be placed (no `elapsed_s`): without ghosts there is no field to finish in |
| `mean_position_regret`, `mean_final_position` | summary, only when some row has a position |
| `ref_models_agree` | optional (`--world-configs`): every alternative world's reference has the same number of stops and a first pit lap within 2 of the world's. `score` then also writes `*_summary_agree.csv`, the table over the agreeing rows |

Row statuses: `ok`, `invalid`, `unavailable` (the policy does not apply here, e.g. the actual
strategy of a driver whose real stint broke the minimum), `error` (the policy itself failed,
e.g. an API error; not counted as invalid), `no_reference` (no legal plan exists).

Each row is stamped with the scenario set version and digest, `belief_config_hash`,
`world_config_hash`, `belief_window`, `sim_code_version`, the seed and `n_samples` (and
`ref_world_config_hashes` with `--world-configs`). Rows scored before the split carry one
`config_hash` instead.

## Current numbers (v1, deterministic)

The current table is `reports/eval/v1_baselines_split.csv` (summary
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

LLM policies stay on the desktop.

## Tests

`tests/test_eval.py` runs offline: synthetic races for sampling and scoring, plus the
committed Spain cache for the v1 set and an end-to-end baseline run.
`tests/test_eval_integration.py` covers the belief and world split and the decision unit,
`tests/test_eval_part2.py` positions, the drawn-events warning, the disagreement flag, the
eval split, `run-all` with two spawned workers, the agent policy, the snapshot contract and
the minimum-stint rule. No test calls a model.
