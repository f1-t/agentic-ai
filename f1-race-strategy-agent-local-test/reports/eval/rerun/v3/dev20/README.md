# dev20 development pass, v3 configuration (2026-11-22, run 2026-11-23 UTC)

Development set only: dev20, 19 development races, 469 scenarios. No held-out race was built, loaded or run.
This run is labelled by its commit and world hash: f53d545 / world 9b98afd55974. It should be repeated
if the safety-car queue's adoption fails its re-measurement.

## What was run

- Commit f53d5458a46dd1ef30fd0ed190a61a7e0fb812ce, the main merge of PR #146, the safety-car queue anchor fix.
  - Tag `eval-v3` points at it (provenance.json `tag`).
  - Tree clean (`dirty: false`).
- SIM_CODE_VERSION 16e496eb8153.
- World: `--world-config default`, config_hash 9b98afd55974. The choices are named in provenance.json `world.choices`:
  - linear_fuel with driver effects, no hinge;
  - the wear cliff with the as-of edge;
  - the thin-fit wear guard;
  - hybrid SC/VSC (history durations, pooled);
  - the safety-car queue (single, record placement, held 1.25 laps);
  - tyre inventory on, no traffic.
- Belief: `--belief-config default` (configs/sim/belief.yaml, file sha256 8e36b0ea7686), config_hash b9891459af63.
  - It adds the thin-fit guard (pace_wear, 3 drivers / 40 rows, sd scaled), tyre inventory, the point cliff edge and as-of pit loss and rates.
  - The agent's tools read the same belief.
- Scenario set: dev20, rule as_of, 25 per race, seed 0.
  - 469 scenarios, 6 skipped (2022 Austria, sets with no recorded compound).
  - Digest a269f07b5b76: the same as the committed reports/eval/rerun/dev20/scenarios.json. Only the sim_code_version stamp differs.
- configs/eval/reporting.yaml sha256 7bf04ee516bd0b271d69759431719e0b60489540b183d98f23e92efb4b01d88a.
- Model and loop settings, from provenance.json:
  - model qwen3-30b-a3b on the voyager backend, VOYAGER_MAX_TOKENS 1024, VOYAGER_TEMPERATURE unset (the backend default);
  - max_steps 10, AGENT_PROTOCOL v2free, AGENT_TURN_REPEAT run. Both are the code defaults; the variables were not set.
  - AGENT_TURN_STATUS, AGENT_REPEAT_MODE, AGENT_SUBMIT_GRACE_STEPS and VOYAGER_TEMPERATURE were unset (checked before every live command). The other loop switches were unset.
- Jobs:
  - references and baselines `--jobs 4`;
  - agent pass `--jobs 10`;
  - timing sample 1 job.
  - OPENBLAS/OMP/MKL threads 1 for every stage.

## Order of the agent stage (differs from the pipeline's built-in order)

The coordinator asked for the full pass first, while another session's job was loading the machine, and the
isolated timing sample afterwards on a quiet machine. The pipeline only runs `--timing-sample` before the pass,
so it was done by hand:

1. `agent --jobs 10` over all 469 scenarios, with no timing sample.
2. Afterwards, the 24 scenarios that `agent_pass.timing_sample(items, 24, seed=0)` picks (the same seeded
   stratified draw `--timing-sample 24` makes; list in run_support/timing_ids.txt) were run with
   `agent --jobs 1 --only timing_ids.txt` into a sibling directory (run_support/timing_sample_pass/, with
   copies of scenarios.json and references/).
3. Its agent.csv was copied in as `timing_sample.csv`, and its trajectories as `timing_sample/trajectories/`
   (its agent_run.json and commands.log in `timing_sample/`). The `tables` stage then built
   tables/agent_latency*.csv with the pipeline's own `agent_timing`.

The results are the main pass's rows. The timing sample supplies only the latency figure and the paired
decision check. agent_run.json in this directory records only the main pass. The timing pass's record is
run_support/timing_sample_pass/agent_run.json.

`tables` was run twice. The first run was without the model environment, so provenance.json had no model id
and the lint flagged it. The second run used the same environment as the agent stage (key from .env,
LLM_BACKEND=voyager, no model calls in this stage). The tables are deterministic. commands.log lists both runs.

## Wall times

| stage | time |
| --- | --- |
| scenarios | 37 s |
| references | 153 s |
| baselines | 185 s |
| agent pass, 469 episodes, 10 jobs | 717 s (03:37:22-03:49:19 UTC) |
| timing sample, 24 episodes, 1 job | 265 s (03:49:41-03:54:07 UTC) |
| tables --harness | 44 s (harness default part: 43 passed, 57 deselected) |
| lint (python -m src.eval.smoke) | 61 s, 0 findings |

## Load during the timing sample

- No other evaluation or test process ran. The other session's src.eval jobs had ended at 03:49 UTC.
- A desktop game (Palworld) was using about 3.5 cores.
- The 1-minute load average was 6.0 to 9.1 on 16 threads (logs/load.log, every 30 s).
- So the tools' local computation did not run on an idle machine.

## Before any model call (gate)

- At f53d545, baselines.csv had no error or invalid row, and 0 rows with negative regret (minimum 0.0).
- The batch probe (a scratch script outside the repository (v3_batch_probe.py)) gave identical per-draw
  times for the reference plan scored alone and inside the reference search: 0/50 draws differing on
  2022_09_canada:ALO:25 and 2022_21_brazil:GAS:35.
- The same check at d35911e (before #146) failed. 4/50 draws differed by up to 98.8 s, and 35 baseline rows had
  negative regret. Those baselines are in run_support/d35911e_queue_defect/.

## Pre-flight

- d35911e:
  - fast tier 1448 passed;
  - slow tier stopped by hand after 46 passed, 0 failed, at 20.6 min, to check out f53d545;
  - live smoke clean (logs/smoke.log).
- f53d545: fast tier 1452 passed (logs/pytest_fast_f53.txt). The slow tier was not run at f53d545.

## Commands (from f1-race-strategy-agent-local-test, D = this directory's scratch original)

    python -m src.eval.rerun scenarios  --set dev20 --out-dir D
    python -m src.eval.rerun references --set dev20 --out-dir D --world-config default --jobs 4
    python -m src.eval.rerun baselines  --set dev20 --out-dir D --world-config default --belief-config default --jobs 4
    python -m src.eval.rerun agent      --set dev20 --out-dir D --world-config default --belief-config default --jobs 10
    python -m src.eval.rerun agent      --set dev20 --out-dir D_timing --world-config default --belief-config default --jobs 1 --only timing_ids.txt
    # D_timing/agent.csv -> D/timing_sample.csv; D_timing/trajectories -> D/timing_sample/trajectories
    python -m src.eval.rerun tables     --set dev20 --out-dir D --harness
    python -m src.eval.smoke D

## Extra analysis (not pipeline output)

The analysis scripts are in run_support/: analyse.py, departures.py and view.py, with their output analysis.txt
and dev20_f53_departures.csv.

- Departures are the agent's scored plan against the last `rank_plans` plan 1, which equalled belief_optimizer's
  plan in 469/469.
- "Helped" or "hurt" is agent minus belief_optimizer regret beyond 0.05 s.
- The reasons are a keyword classification of the stated rationale.
- "No circuit history" means every tyre-life source shown by `race_picture` in that race is a generic tier (3 or higher).
