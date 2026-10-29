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
whatever `LLM_BACKEND` is configured.

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

Next versions: v2 once Sarah's snapshot builder and the new races land (pass it as
`builder=` to `build_set`; it has the same signature as `snapshot_at`).

## Metrics

For each scenario the optimizer (`optimizer_bridge.optimize_batch`, Quan's search scored by
Jason's simulator) finds the best plan with at most `max_stops` future stops. Every policy's
plan is scored by the same simulator under the same draws (docs/rules.md R6).

| column | meaning |
| --- | --- |
| `regret_s` | mean remaining time of the policy's plan minus the optimizer's. A plan with more stops than the optimizer searches can come in below it, so this can be negative. |
| `regret_pct` | `regret_s` as a percentage of the optimizer's remaining time |
| `invalid_rate` | invalid plans / (ok + invalid). Invalid: a rule broken (docs/rules.md) or no plan submitted. |
| `stop_agreement` | share of scored plans with the optimizer's number of future stops |
| `first_pit_error` | \|first pit lap - optimizer's first pit lap\|; 0 when neither stops, blank when only one does |
| `latency_s`, `tool_calls`, `input_tokens`, `output_tokens` | per decision, when the policy reports them |

Row statuses: `ok`, `invalid`, `unavailable` (the policy does not apply here, e.g. the actual
strategy of a driver whose real stint broke the minimum), `error` (the policy itself failed,
e.g. an API error; not counted as invalid), `no_reference` (no legal plan exists).

Each row is stamped with the scenario set version and digest, the simulator's `config_hash`
and `sim_code_version`, the seed and `n_samples`.

## Current numbers (v1, default SimConfig, full fit window, deterministic)

Overall, from `reports/eval/v1_baselines_summary.csv`:

| policy | scored | mean regret s | median | p90 | stop agreement |
| --- | --- | --- | --- | --- | --- |
| fixed_one_stop | 250 | 4.95 | 3.01 | 13.60 | 0.90 |
| actual_strategy | 199 | 6.72 | 7.11 | 14.52 | 0.57 |
| fixed_two_stop | 211 | 12.95 | 9.25 | 21.18 | 0.00 |
| greedy_crossover | 123 | 18.77 | 18.61 | 23.85 | 0.06 |
| baseline agent (recorded, 1 scenario) | 1 | 6.29 | | | 0.00 |

The baseline agent row is the recorded run in `docs/sample_run.txt` (HAM from lap 0), scored
under this simulator rather than the legacy linear one it was graded on. It is the only
scenario the current agent loop can answer; the live agent covers the set once the loop takes
a snapshot.

Things the numbers show that the owners may want to look at:
- `greedy_crossover` stops immediately (a zero-lap first stint, R1) on 141 of 264 scenarios,
  so it is unavailable there.
- `actual_strategy` is unavailable on 65 scenarios where a real stint was shorter than the
  10-lap minimum.

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

In `slurm/eval_chunk.sbatch` that means replacing `python -m src.sim.jobs eval_chunk ...` with
the `python -m src.eval jobs ... -- eval_chunk ...` form.

Use the per-task form (or `n_jobs: 1`): `run-all` with worker processes does not carry the
runner into the workers. LLM policies stay on the desktop.

## Tests

`tests/test_eval.py` runs offline: synthetic races for sampling and scoring, plus the
committed Spain cache for the v1 set and an end-to-end baseline run.
