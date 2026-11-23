# test9 held-out rerun, v3 configuration (2026-11-22, run 2026-11-23 UTC)

The owner ordered this test-set rerun under the v3 configuration, and the coordinator gave the GO.
Label: f53d545 (tag `eval-v3`), world 9b98afd55974.

## What was run

- Commit f53d5458a46dd1ef30fd0ed190a61a7e0fb812ce, tag `eval-v3`, tree clean (provenance `dirty: false`).
- SIM_CODE_VERSION 16e496eb8153.
- World `--world-config default`, config_hash 9b98afd55974.
- Belief `--belief-config default` (belief.yaml sha256 8e36b0ea7686), config_hash b9891459af63.
- The configuration is the same as ../dev20/ (see its README).
- configs/eval/reporting.yaml sha256 7bf04ee516bd0b271d69759431719e0b60489540b183d98f23e92efb4b01d88a. The tables
  stage checked it against the set's recorded value and accepted it.
- Scenario set:
  - The committed reports/eval/rerun/test9/scenarios.json was copied into a fresh out-dir unchanged.
  - 225 scenarios, 9 races x 25, digest 663f59b88d5f, file sha256 fe76246030b7… before and after the run.
  - It still carries the earlier sim_code_version stamp, c8f30186efed.
  - The `scenarios` stage was not run, and `--final` was never passed.
- Model and loop settings, from provenance.json:
  - model qwen3-30b-a3b on the voyager backend, VOYAGER_MAX_TOKENS 1024, temperature unset;
  - max_steps 10, AGENT_PROTOCOL v2free and AGENT_TURN_REPEAT run, both as code defaults (not set);
  - every other loop switch unset. The four forbidden switches were checked before every live command.
- Jobs:
  - references and baselines 4;
  - agent pass 10;
  - timing sample 1.
  - BLAS threads 1 everywhere.

## Gate (before any model call)

- Rows: 9 races x 25 scenarios, 0 error rows, 0 invalid rows.
- Regret: 0 negative (minimum 0.0).
- Unavailable rows: rule or tyre-set reasons only.
- 2022_03_australia:ALB:55 is `no_reference` for every policy ("No legal remaining plan exists within the
  future-stop limit"), as in the previous run.
- Batch probe (a scratch script outside the repository (v3_batch_probe.py)): 0/50 differing draws on
  2020_16_sakhir:BOT:25 and 2022_03_australia:LAT:30.
  - This is the one non-pipeline load of test-set data. The coordinator asked for it.

## Order of the agent stage

As for dev20-v3: the pass ran first at 10 jobs. Afterwards the 24 scenarios of
`agent_pass.timing_sample(items, 24, seed=0)` (list in run_support/t9_timing_ids.txt) were run in a sibling
directory at 1 job with `--only` (run_support/timing_sample_pass/). That run's agent.csv and trajectories
were copied in as `timing_sample.csv` and `timing_sample/`, and the `tables` stage built the latency tables.
`tables` ran once, with the agent's environment, so provenance.json records the model.

## Wall times (UTC)

| stage | time |
| --- | --- |
| references --jobs 4 | 76 s (04:03:38-04:04:55) |
| baselines --jobs 4 | 96 s (04:05:05-04:06:41) |
| agent pass, 225 scenarios, 10 jobs | 382 s (04:07:36-04:13:59) |
| timing sample, 24, 1 job | 220 s (04:14:11-04:17:51) |
| tables --harness | 57 s (harness default part 43 passed, 57 deselected) |
| lint | 30 s, 2 findings |

## Load during the timing sample

- No other evaluation or test process ran.
- A desktop game used about 3.5 cores.
- The 1-minute load average was 6.6 to 11.4 on 16 threads (logs/load.log). It started high because the 10-job
  pass had just finished.

## Lint: 2 findings

Both are the `no_reference` scenario 2022_03_australia:ALB:55, which the agent stage does not run:
"no trajectory" and "no decision on the agent row". The previous run had the same two findings, and a third
(no model id) that this run does not have.

## Commands (from f1-race-strategy-agent-local-test; D = the scratch original of this directory)

    cp reports/eval/rerun/test9/scenarios.json D/
    python -m src.eval.rerun references --set test9 --out-dir D --world-config default --jobs 4
    python -m src.eval.rerun baselines  --set test9 --out-dir D --world-config default --belief-config default --jobs 4
    python -m src.eval.rerun agent      --set test9 --out-dir D --world-config default --belief-config default --jobs 10
    python -m src.eval.rerun agent      --set test9 --out-dir D_timing --world-config default --belief-config default --jobs 1 --only t9_timing_ids.txt
    # D_timing/agent.csv -> D/timing_sample.csv; D_timing/trajectories -> D/timing_sample/trajectories
    python -m src.eval.rerun tables     --set test9 --out-dir D --harness
    python -m src.eval.smoke D

## Extra analysis

run_support/ holds the scripts, their output (t9_analysis.txt) and test9_departures.csv. They read only this
pass's output files, not race data. Definitions are as in the dev20-v3 README.
