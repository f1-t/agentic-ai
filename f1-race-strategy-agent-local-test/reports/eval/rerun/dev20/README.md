# dev20 at eval-v2-dev2: the final development pass

Development set only: dev20, the 19 calibration races that are ever sampled, 469 scenarios. These
are races the simulators were calibrated on. No held-out race was built, loaded or run. The
results and how they were computed are in docs/eval.md, "Development result (dev20) at
eval-v2-dev2".

- Tag `eval-v2-dev2`, commit 9770eb54ff2f8923712da44cc90aefebf247030b, tree clean
  (`provenance.json`: `dirty: false`).
- SIM_CODE_VERSION c8f30186efed. World `--world-config default`, config_hash 5069228b35e7. Belief
  `--belief-config default` (configs/sim/belief.yaml), config_hash 351f4124ff22; the agent's tools
  read the same belief.
- Scenario set dev20, rule as_of, 25 per race, seed 0, digest a269f07b5b76 (6 sampled points in
  2022 Austria skipped: no recorded compound).
- configs/eval/reporting.yaml sha256 7bf04ee516bd0b271d69759431719e0b60489540b183d98f23e92efb4b01d88a.
- Model qwen3-30b-a3b (voyager backend, VOYAGER_MAX_TOKENS 1024, temperature at the backend
  default). Loop: max_steps 10, AGENT_TURN_REPEAT=run; the other loop switches unset.
- references and baselines at `--jobs 4`; the agent: a timing sample of 24 run one at a time, then
  the pass at `--jobs 10`.

## Commands (from f1-race-strategy-agent-local-test, LLM_BACKEND=voyager, key from .env)

    python -m src.eval.rerun scenarios  --set dev20 --out-dir D
    python -m src.eval.rerun references --set dev20 --out-dir D --world-config default --jobs 4
    python -m src.eval.rerun baselines  --set dev20 --out-dir D --world-config default --belief-config default --jobs 4
    AGENT_TURN_REPEAT=run python -m src.eval.rerun agent --set dev20 --out-dir D --world-config default --belief-config default --timing-sample 24 --jobs 10
    python -m src.eval.rerun tables --set dev20 --out-dir D --harness
    python -m src.eval.smoke D

| stage | wall time |
| --- | --- |
| scenarios | 32 s |
| references | 62 s |
| baselines | 74 s |
| agent, timing sample | 828 s (24 episodes, 1 job, 11:31:29-11:45:16 UTC) |
| agent, pass | 2660 s (469 episodes, 10 jobs, 11:45:16-12:29:36 UTC) |
| tables --harness | 18 s (future-blindness harness default part: 38 passed) |
| lint | 51 s, 0 findings |

## What is here

| path | what |
| --- | --- |
| `scenarios.json` | the 469 scenarios |
| `references/dev20/` | the world's references (the cache the scoring read) |
| `baselines.csv` | every non-LLM policy's rows (3283) |
| `agent.csv`, `trajectories/` | the agent's 469 rows and one trajectory per scenario |
| `timing_sample.csv`, `timing_sample/trajectories/` | the isolated timing sample (timing only; the pass ran those scenarios again) |
| `agent_run.json` | the two passes: times, model and loop switches |
| `tables/` | combined rows, summaries, paired and tail tables, latency tables, the pre-registered `reporting_*` tables |
| `provenance.json` | commit, tag, configs and hashes, SIM_CODE_VERSION, model, loop, reporting.yaml verbatim, commands |

Left out: `parts/` (per-race and per-worker intermediates the stages merge into the files above)
and the logs. The run's output directory was a scratch path; in `provenance.json` it is written
`<out-dir>` in the commands, and the belief config path is written relative to the project. Nothing
else in the files was changed.

To check the tables: copy this directory without `tables/`, run
`python -m src.eval.rerun tables --set dev20 --out-dir <copy>` (no model calls; it reads the rows)
and compare; every table regenerates byte-identical. Run it on a copy: the stage rewrites
`provenance.json` with the commit it runs at.
