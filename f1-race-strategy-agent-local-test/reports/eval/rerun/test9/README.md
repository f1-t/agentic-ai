# test9 at eval-v2-test: the one-time held-out run (2026-11-22)

**Gate held. The pass completed cleanly.** 225/225 scenarios: 224 submitted and scored, 0 invalid,
0 errors, 0 API retries. The one remaining scenario, 2022_03_australia:ALB:55, is `no_reference`
for every policy: on HARD for 55 laps with 3 to go and no stop yet, no legal plan exists.

**Pre-registered verdicts, fresh five** (2021 France, 2021 Mexico, 2022 Miami, 2022 Australia,
2022 Belgium; agent minus X; negative means the agent is better; 95% (race, driver) interval, race
interval beside):

| X | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |
| --- | --- | --- | --- | --- | --- |
| run_to_flag (lead) | +1.59 | (-0.86, +4.05) | (-5.20, +7.24) | 33 / 40 / 51 | **level** |
| one_stop_total (lead) | +1.53 | (-1.01, +4.08) | (-5.73, +7.50) | 31 / 43 / 50 | **level** |
| actual_strategy | +2.33 | (-0.10, +4.64) | (-2.18, +8.50) | 31 / 39 / 46 | **level** |
| belief_optimizer | -0.04 | (-0.54, +0.36) | (-0.69, +0.38) | 34 / 61 / 29 | **level** |

On the fresh five the agent is not shown to beat the simple rules. It is numerically behind them
by about 1.6 s, but the interval includes zero. On the earlier four it is ahead of both lead rules
(-5.44 and -5.38 s). Over all nine it is level with them (-1.55 and -1.56 s, intervals reaching
+0.33 and +0.36). Against the teams' actual strategy it is ahead over all nine (-4.55 s). In every
group it is level with belief_optimizer, the optimizer its tools run.

**Where the fresh-five gap comes from.** It sits in two races. Agent minus run_to_flag per race:

| race | agent - run_to_flag |
| --- | --- |
| 2022 Australia | +11.61 s |
| 2022 Miami | +6.21 s |
| 2021 France | +0.87 s |
| 2021 Mexico | +0.69 s |
| 2022 Belgium | -11.02 s |

belief_optimizer has the same shortfall in those races (Australia 19.31, Miami 9.35, against the
agent's 17.96 and 9.39). In Australia and Miami the belief plans a two-stop, MEDIUM then SOFT or
HARD, where the world's optimum is a one-stop with a long HARD stint. The agent's regret equals
belief_optimizer's on 106 of 224 scenarios. So the loss to the rules there comes from the belief and tools, not
from the language model.

## Run

- Tag `eval-v2-test`, commit 82f06218989fd0f989734f93b2af281a5a81209c. Detached in a fresh worktree;
  `git status --porcelain` was empty before, during and after the run; provenance `dirty: false`.
  Nothing was committed or pushed. `cache/fastf1_http_cache.sqlite` was not modified.
- SIM_CODE_VERSION c8f30186efed. World `--world-config default`, config_hash 5069228b35e7. Belief
  `--belief-config default`, config_hash 351f4124ff22. The stamps on every row match.
- `configs/eval/reporting.yaml` sha256 7bf04ee516bd0b271d69759431719e0b60489540b183d98f23e92efb4b01d88a:
  file, set and provenance agree, and the tables stage accepted it.
- Set test9: 225 scenarios (9 races x 25), 0 skipped, digest 663f59b88d5f, rule as_of, seed 0,
  development false. It was built earlier (12:26 local) and not rebuilt. Its sha256 (fe762460...)
  was unchanged after the run.
- Model qwen3-30b-a3b, voyager backend, VOYAGER_MAX_TOKENS 1024, temperature at the backend
  default. Loop: max_steps 10, AGENT_TURN_REPEAT=run. AGENT_TURN_STATUS, AGENT_REPEAT_MODE,
  AGENT_SUBMIT_GRACE_STEPS and VOYAGER_TEMPERATURE were checked unset before the agent stage.
- Jobs: references 4, baselines 4. Agent: a timing sample of 24 at 1 job, then the pass at 10 jobs.
  OPENBLAS/OMP/MKL threads were 1 for every stage.

| stage | wall time (UTC) |
| --- | --- |
| references --jobs 4 | 34 s (19:29:13-19:29:48) |
| baselines --jobs 4 | 41 s (19:29:56-19:30:38) |
| agent, timing sample (24, 1 job) | 750 s (19:31:48-19:44:17) |
| agent, pass (225, 10 jobs) | 1256 s (19:44:17-20:05:13); the stage in all 2005 s |
| tables --harness | 35 s (future-blindness default part: 38 passed, 57 deselected; 7 fixed leaks, 0 known) |
| lint (`src.eval.smoke`) | 31 s, 3 findings (below) |

**Commands** (from `f1-race-strategy-agent-local-test/`, OUT = this directory's scratch original):

    python -m src.eval.rerun references --set test9 --out-dir OUT --world-config default --jobs 4
    python -m src.eval.rerun baselines  --set test9 --out-dir OUT --world-config default --belief-config default --jobs 4
    AGENT_TURN_REPEAT=run LLM_BACKEND=voyager (key from .env) \
    python -m src.eval.rerun agent      --set test9 --out-dir OUT --world-config default --belief-config default --timing-sample 24 --jobs 10
    python -m src.eval.rerun tables     --set test9 --out-dir OUT --harness
    python -m src.eval.smoke OUT

**Machine load during the timing sample:** the 1-minute load average had a median of 0.67 (range
0.19 to 1.17) on 16 threads, sampled every 30 s (`run_support/load.log`). No other eval or test
process ran, and the top non-agent processes were desktop apps under 20% CPU. Before the run there
were orphaned, idle multiprocessing workers from another worktree's run 11.5 h earlier, at 0% CPU.
During the pass the median load was 4.0 (max 9.3).

**Gate (before any model call), passed:**
- all nine races with 25 scenarios each;
- 1575 baseline rows, 0 error rows, 0 invalid rows;
- 0 negative regrets among 1348 scored rows (the minimum is 0.0);
- belief_optimizer, run_to_flag and one_stop_total each have a row for all 225, with 0 unavailable
  and 1 no_reference (ALB:55, as above).

The unavailable rows, all published-rule reasons:
- actual_strategy 11 (5 new stint below the minimum, 1 current stint below the minimum before a
  stop, 5 stop immediately, R1);
- fixed_one_stop 7 (stop immediately);
- fixed_two_stop 13 (too few laps for 2 stops);
- greedy_crossover 189 (stop immediately).

**Things to know about this run (none changes a result):**
1. **Lint: 3 findings.**
   - Two are the `no_reference` scenario ALB:55: "no trajectory" and "no decision on the agent row".
     The agent is not run where no legal plan exists.
   - The third is "provenance: no model id (ANTHROPIC_MODEL)". `provenance.json`'s top-level
     `model` block is filled from the tables stage's own environment, and I ran tables without
     `.env` or LLM_BACKEND, so the block is null and the lint fell back to the anthropic key. The
     model id and the loop settings are recorded in the same file under `agent_run.passes[*].env`
     (voyager, qwen3-30b-a3b, 1024, AGENT_TURN_REPEAT=run). dev20 ran tables with the env loaded.
     The tables were not rerun (each stage runs once).
2. **The first references launch never ran Python.** A command mixing the stage with `git status`
   was refused by the worktree guard ("names git in a form too complex to verify"). The retry used
   `/usr/bin/time`, which is not installed: exit 127 before Python started, and nothing was written
   but a one-line log, later overwritten. The stage then ran once.
3. I printed the non-secret values of VOYAGER_MODEL and VOYAGER_MAX_TOKENS to my console once, to
   confirm the settings. No key, token or endpoint was printed. Key grep before copying: 0 of 316
   files hold the key, 0 its 12-character prefix, 0 the endpoint.
4. Each stage logged pit-loss-prior warnings: 2021_02_emilia_romagna ("data ... not loaded yet")
   and 2021_12_belgium ("None of ['driver'] are in the columns") were left out of the pit-loss
   prior. Both are earlier races, not held-out ones. The scenarios stage logged the Belgium one too.

# Tables (generated from tables/ by run_support/make_tables.py; dev20 from reports/eval/rerun/dev20/tables at the tag)

### PAIRED

| group | X | paired (clusters) | agent mean | X mean | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fresh five | run_to_flag | 124 (72) | 7.44 | 5.85 | +1.59 | (-0.86, +4.05) | (-5.20, +7.24) | 33 / 40 / 51 | level |
| fresh five | one_stop_total | 124 (72) | 7.44 | 5.91 | +1.53 | (-1.01, +4.08) | (-5.73, +7.50) | 31 / 43 / 50 | level |
| fresh five | actual_strategy | 116 (69) | 7.28 | 4.95 | +2.33 | (-0.10, +4.64) | (-2.18, +8.50) | 31 / 39 / 46 | level |
| fresh five | belief_optimizer | 124 (72) | 7.44 | 7.49 | -0.04 | (-0.54, +0.36) | (-0.69, +0.38) | 34 / 61 / 29 | level |
| fresh five | fixed_one_stop | 121 (71) | 7.63 | 10.87 | -3.24 | (-5.75, -0.68) | (-8.73, +0.95) | 66 / 5 / 50 | ahead |
| fresh five | fixed_two_stop | 118 (70) | 7.82 | 26.54 | -18.72 | (-21.43, -16.02) | (-21.63, -15.61) | 95 / 0 / 23 | ahead |
| fresh five | greedy_crossover | 20 (19) | 12.31 | 66.35 | -54.04 | (-85.31, -28.23) | (-82.93, -25.05) | 20 / 0 / 0 | ahead |
| earlier four | run_to_flag | 100 (61) | 5.13 | 10.57 | -5.44 | (-8.56, -2.98) | (-11.91, +0.15) | 50 / 29 / 21 | ahead |
| earlier four | one_stop_total | 100 (61) | 5.13 | 10.51 | -5.38 | (-8.52, -2.92) | (-11.91, +0.21) | 50 / 29 / 21 | ahead |
| earlier four | actual_strategy | 97 (59) | 4.97 | 17.75 | -12.78 | (-19.33, -7.29) | (-30.38, -2.07) | 51 / 28 / 18 | ahead |
| earlier four | belief_optimizer | 100 (61) | 5.13 | 5.08 | +0.05 | (-0.19, +0.28) | (-0.13, +0.24) | 25 / 45 / 30 | level |
| earlier four | fixed_one_stop | 96 (60) | 5.34 | 15.01 | -9.66 | (-12.72, -7.07) | (-20.06, -3.58) | 76 / 0 / 20 | ahead |
| earlier four | fixed_two_stop | 93 (59) | 5.52 | 27.65 | -22.14 | (-26.70, -18.18) | (-33.62, -13.78) | 85 / 0 / 8 | ahead |
| earlier four | greedy_crossover | 15 (15) | 10.48 | 81.67 | -71.19 | (-93.97, -47.50) | (-114.71, -38.47) | 15 / 0 / 0 | ahead |
| all nine | run_to_flag | 224 (133) | 6.41 | 7.96 | -1.55 | (-3.50, +0.33) | (-6.53, +3.28) | 83 / 69 / 72 | level |
| all nine | one_stop_total | 224 (133) | 6.41 | 7.97 | -1.56 | (-3.57, +0.36) | (-6.74, +3.40) | 81 / 72 / 71 | level |
| all nine | actual_strategy | 213 (128) | 6.23 | 10.78 | -4.55 | (-8.08, -1.38) | (-14.70, +2.83) | 82 / 67 / 64 | ahead |
| all nine | belief_optimizer | 224 (133) | 6.41 | 6.41 | -0.00 | (-0.30, +0.24) | (-0.36, +0.27) | 59 / 106 / 59 | level |
| all nine | fixed_one_stop | 217 (131) | 6.62 | 12.70 | -6.08 | (-8.05, -4.23) | (-11.79, -1.70) | 142 / 5 / 70 | ahead |
| all nine | fixed_two_stop | 211 (129) | 6.81 | 27.03 | -20.23 | (-22.64, -17.84) | (-25.64, -16.25) | 180 / 0 / 31 | ahead |
| all nine | greedy_crossover | 35 (34) | 11.53 | 72.92 | -61.39 | (-81.95, -42.76) | (-85.86, -38.48) | 35 / 0 / 0 | ahead |

### BY ERA

| group | X | paired (clusters) | agent mean | X mean | agent - X | 95% CI (race, driver) | 95% CI (race) | W / T / L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-20 (2) | run_to_flag | 50 (31) | 3.58 | 9.89 | -6.31 | (-11.74, -1.87) | (-14.78, +2.16) | 22 / 15 / 13 | ahead |
| 2019-20 (2) | one_stop_total | 50 (31) | 3.58 | 9.89 | -6.31 | (-11.74, -1.87) | (-14.78, +2.16) | 22 / 15 / 13 | ahead |
| 2019-20 (2) | actual_strategy | 48 (30) | 3.56 | 27.15 | -23.59 | (-34.94, -13.80) | (-39.29, -6.54) | 28 / 13 / 7 | ahead |
| 2019-20 (2) | belief_optimizer | 50 (31) | 3.58 | 3.44 | +0.14 | (-0.19, +0.50) | (-0.06, +0.34) | 13 / 24 / 13 | level |
| 2021 (2) | run_to_flag | 50 (29) | 3.87 | 3.09 | +0.78 | (-1.32, +2.84) | (+0.69, +0.87) | 12 / 16 / 22 | level |
| 2021 (2) | one_stop_total | 50 (29) | 3.87 | 2.82 | +1.05 | (-1.16, +3.11) | (+0.69, +1.41) | 10 / 18 / 22 | level |
| 2021 (2) | actual_strategy | 50 (29) | 3.87 | 3.98 | -0.11 | (-2.31, +1.82) | (-2.87, +2.65) | 14 / 16 / 20 | level |
| 2021 (2) | belief_optimizer | 50 (29) | 3.87 | 3.63 | +0.25 | (-0.02, +0.55) | (+0.24, +0.26) | 9 / 25 / 16 | level |
| 2022 (3) | run_to_flag | 74 (43) | 9.86 | 7.71 | +2.14 | (-1.73, +6.13) | (-11.02, +11.62) | 21 / 24 / 29 | level |
| 2022 (3) | one_stop_total | 74 (43) | 9.86 | 8.01 | +1.85 | (-2.17, +6.00) | (-12.14, +11.73) | 21 / 25 / 28 | level |
| 2022 (3) | actual_strategy | 66 (40) | 9.86 | 5.68 | +4.18 | (+0.39, +7.83) | (-3.51, +14.82) | 17 / 23 / 26 | behind |
| 2022 (3) | belief_optimizer | 74 (43) | 9.86 | 10.09 | -0.24 | (-1.02, +0.41) | (-1.34, +0.53) | 25 / 36 / 13 | level |
| 2024-25 (2) | run_to_flag | 50 (30) | 6.68 | 11.26 | -4.58 | (-7.05, -2.52) | (-5.88, -3.29) | 28 / 14 / 8 | ahead |
| 2024-25 (2) | one_stop_total | 50 (30) | 6.68 | 11.14 | -4.46 | (-6.91, -2.40) | (-5.64, -3.29) | 28 / 14 / 8 | ahead |
| 2024-25 (2) | actual_strategy | 49 (29) | 6.36 | 8.55 | -2.19 | (-3.38, -0.90) | (-3.68, -0.64) | 23 / 15 / 11 | ahead |
| 2024-25 (2) | belief_optimizer | 50 (30) | 6.68 | 6.73 | -0.05 | (-0.43, +0.20) | (-0.21, +0.12) | 12 / 21 / 17 | level |

### COUNTS (scored / invalid / unavailable / error / no_reference)

| policy | fresh five | earlier four | all nine |
| --- | --- | --- | --- |
| agent | 124 / 0 / 0 / 0 / 1 | 100 / 0 / 0 / 0 / 0 | 224 / 0 / 0 / 0 / 1 |
| belief_optimizer | 124 / 0 / 0 / 0 / 1 | 100 / 0 / 0 / 0 / 0 | 224 / 0 / 0 / 0 / 1 |
| run_to_flag | 124 / 0 / 0 / 0 / 1 | 100 / 0 / 0 / 0 / 0 | 224 / 0 / 0 / 0 / 1 |
| one_stop_total | 124 / 0 / 0 / 0 / 1 | 100 / 0 / 0 / 0 / 0 | 224 / 0 / 0 / 0 / 1 |
| actual_strategy | 116 / 0 / 8 / 0 / 1 | 97 / 0 / 3 / 0 / 0 | 213 / 0 / 11 / 0 / 1 |
| fixed_one_stop | 121 / 0 / 3 / 0 / 1 | 96 / 0 / 4 / 0 / 0 | 217 / 0 / 7 / 0 / 1 |
| fixed_two_stop | 118 / 0 / 6 / 0 / 1 | 93 / 0 / 7 / 0 / 0 | 211 / 0 / 13 / 0 / 1 |
| greedy_crossover | 20 / 0 / 104 / 0 / 1 | 15 / 0 / 85 / 0 / 0 | 35 / 0 / 189 / 0 / 1 |

### REGRET mean / median / p90 / worst (s), scored n

| race | agent | belief_optimizer | run_to_flag | one_stop_total | actual_strategy | fixed_one_stop | fixed_two_stop | greedy_crossover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019_06_monaco | 3.5 / 0.2 / 15.0 / 20.9 (n 25) | 3.1 / 0.0 / 15.0 / 21.1 (n 25) | 18.2 / 12.3 / 41.3 / 54.0 (n 25) | 18.2 / 12.3 / 41.3 / 54.0 (n 25) | 42.8 / 44.4 / 89.1 / 114.5 (n 25) | 30.1 / 33.5 / 45.9 / 54.0 (n 21) | 42.9 / 42.7 / 57.8 / 63.1 (n 21) | 54.0 / 65.5 / 72.6 / 74.3 (n 3) |
| 2020_16_sakhir | 3.7 / 2.8 / 7.9 / 13.1 (n 25) | 3.7 / 2.4 / 8.1 / 13.9 (n 25) | 1.5 / 0.4 / 3.9 / 7.8 (n 25) | 1.5 / 0.4 / 3.9 / 7.8 (n 25) | 10.2 / 3.0 / 27.9 / 33.4 (n 23) | 5.7 / 2.4 / 15.8 / 18.2 (n 25) | 25.9 / 24.7 / 39.6 / 73.5 (n 25) | 36.9 / 33.9 / 49.4 / 59.0 (n 5) |
| 2021_07_france | 4.0 / 3.2 / 9.4 / 10.0 (n 25) | 3.7 / 3.3 / 8.5 / 9.2 (n 25) | 3.1 / 0.9 / 12.4 / 18.0 (n 25) | 2.6 / 0.5 / 11.0 / 18.0 (n 25) | 1.3 / 0.2 / 4.6 / 8.3 (n 25) | 4.8 / 1.0 / 14.9 / 18.3 (n 22) | 19.2 / 16.0 / 34.7 / 37.8 (n 22) | 48.3 / 47.8 / 66.2 / 70.8 (n 4) |
| 2021_18_mexico | 3.8 / 1.0 / 9.6 / 17.6 (n 25) | 3.5 / 2.2 / 9.1 / 14.6 (n 25) | 3.1 / 1.2 / 6.7 / 34.1 (n 25) | 3.1 / 1.2 / 6.7 / 34.1 (n 25) | 6.6 / 1.6 / 20.4 / 32.8 (n 25) | 7.3 / 6.2 / 16.0 / 22.5 (n 25) | 24.2 / 27.3 / 38.3 / 43.8 (n 25) | 78.7 / 78.7 / 107.3 / 114.4 (n 2) |
| 2022_03_australia | 18.0 / 21.8 / 25.7 / 28.5 (n 24) | 19.3 / 23.2 / 25.0 / 30.7 (n 24) | 6.3 / 7.7 / 16.5 / 18.8 (n 24) | 6.2 / 5.1 / 16.5 / 18.8 (n 24) | 2.4 / 0.0 / 4.5 / 17.5 (n 21) | 14.6 / 17.7 / 19.0 / 20.6 (n 24) | 32.8 / 38.2 / 39.0 / 39.6 (n 24) | 121.7 / 40.9 / 252.0 / 252.0 (n 5) |
| 2022_05_miami | 9.4 / 0.0 / 27.8 / 28.2 (n 25) | 9.3 / 0.0 / 27.6 / 28.7 (n 25) | 3.2 / 0.2 / 15.3 / 23.7 (n 25) | 3.0 / 0.2 / 14.3 / 23.7 (n 25) | 8.2 / 0.7 / 26.2 / 27.6 (n 24) | 11.0 / 18.9 / 21.6 / 23.0 (n 25) | 33.3 / 39.9 / 42.9 / 47.4 (n 25) | 34.8 / 28.5 / 46.1 / 53.7 (n 4) |
| 2022_14_belgium | 2.5 / 0.0 / 9.7 / 24.2 (n 25) | 2.0 / 0.1 / 9.1 / 15.9 (n 25) | 13.6 / 20.7 / 23.5 / 26.5 (n 25) | 14.7 / 20.8 / 26.4 / 27.9 (n 25) | 6.1 / 7.0 / 15.0 / 15.8 (n 21) | 16.1 / 13.8 / 26.1 / 27.9 (n 25) | 22.0 / 21.5 / 30.0 / 32.7 (n 22) | 45.8 / 24.8 / 92.2 / 131.8 (n 5) |
| 2024_11_austria | 10.0 / 8.6 / 21.1 / 22.6 (n 25) | 10.3 / 8.4 / 20.9 / 22.6 (n 25) | 13.3 / 16.0 / 19.9 / 26.0 (n 25) | 13.3 / 16.0 / 19.9 / 26.0 (n 25) | 10.2 / 8.9 / 17.2 / 18.9 (n 24) | 15.1 / 15.9 / 16.7 / 21.3 (n 25) | 21.0 / 21.6 / 31.5 / 32.4 (n 25) | 104.8 / 130.1 / 130.1 / 130.1 (n 4) |
| 2025_11_austria | 3.3 / 0.0 / 14.6 / 16.6 (n 25) | 3.2 / 0.1 / 14.9 / 16.8 (n 25) | 9.2 / 10.5 / 20.8 / 22.8 (n 25) | 9.0 / 10.5 / 18.5 / 22.5 (n 25) | 7.0 / 7.4 / 15.3 / 19.5 (n 25) | 11.4 / 11.5 / 16.3 / 16.9 (n 25) | 22.7 / 23.8 / 30.9 / 35.5 (n 22) | 153.2 / 152.9 / 153.7 / 153.9 (n 3) |
| ALL | 6.4 / 1.8 / 22.2 / 28.5 (n 224) | 6.4 / 1.4 / 22.6 / 30.7 (n 224) | 8.0 / 1.5 / 22.1 / 54.0 (n 224) | 8.0 / 1.3 / 23.5 / 54.0 (n 224) | 10.8 / 3.2 / 24.9 / 114.5 (n 213) | 12.7 / 12.1 / 21.9 / 54.0 (n 217) | 27.0 / 25.2 / 41.8 / 73.5 (n 211) | 72.9 / 40.9 / 152.9 / 252.0 (n 35) |

Pooled mean with 95% (race, driver) interval: agent 6.41 (5.27, 7.59); belief_optimizer 6.41 (5.22, 7.63); run_to_flag 7.96 (6.42, 9.62); one_stop_total 7.97 (6.38, 9.65); actual_strategy 10.78 (7.81, 14.20); fixed_one_stop 12.70 (11.15, 14.19); fixed_two_stop 27.03 (25.03, 29.11); greedy_crossover 72.92 (53.55, 94.73)

### TAIL

| group | policy | scored | over 20 s (stated) | over 10 s | over 30 s | median s | worst s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fresh five | agent | 124 | 0.194 | 0.266 | 0.000 | 0.71 | 28.46 |
| fresh five | belief_optimizer | 124 | 0.202 | 0.274 | 0.008 | 0.86 | 30.66 |
| fresh five | run_to_flag | 124 | 0.137 | 0.250 | 0.008 | 0.86 | 34.14 |
| fresh five | one_stop_total | 124 | 0.137 | 0.258 | 0.008 | 0.76 | 34.14 |
| fresh five | actual_strategy | 116 | 0.078 | 0.181 | 0.009 | 0.53 | 32.76 |
| fresh five | fixed_one_stop | 121 | 0.165 | 0.545 | 0.000 | 10.56 | 27.86 |
| fresh five | fixed_two_stop | 118 | 0.729 | 0.924 | 0.398 | 25.69 | 47.43 |
| fresh five | greedy_crossover | 20 | 0.950 | 1.000 | 0.600 | 37.26 | 252.02 |
| earlier four | agent | 100 | 0.040 | 0.170 | 0.000 | 2.20 | 22.64 |
| earlier four | belief_optimizer | 100 | 0.040 | 0.180 | 0.000 | 1.74 | 22.64 |
| earlier four | run_to_flag | 100 | 0.170 | 0.450 | 0.100 | 6.99 | 53.99 |
| earlier four | one_stop_total | 100 | 0.160 | 0.450 | 0.100 | 6.99 | 53.99 |
| earlier four | actual_strategy | 97 | 0.216 | 0.454 | 0.144 | 8.60 | 114.47 |
| earlier four | fixed_one_stop | 96 | 0.177 | 0.698 | 0.135 | 13.31 | 53.99 |
| earlier four | fixed_two_stop | 93 | 0.688 | 0.882 | 0.387 | 24.70 | 73.53 |
| earlier four | greedy_crossover | 15 | 1.000 | 1.000 | 0.733 | 65.53 | 153.89 |
| all nine | agent | 224 | 0.125 | 0.223 | 0.000 | 1.77 | 28.46 |
| all nine | belief_optimizer | 224 | 0.129 | 0.232 | 0.004 | 1.37 | 30.66 |
| all nine | run_to_flag | 224 | 0.152 | 0.339 | 0.049 | 1.53 | 53.99 |
| all nine | one_stop_total | 224 | 0.147 | 0.344 | 0.049 | 1.30 | 53.99 |
| all nine | actual_strategy | 213 | 0.141 | 0.305 | 0.070 | 3.21 | 114.47 |
| all nine | fixed_one_stop | 217 | 0.171 | 0.613 | 0.060 | 12.14 | 53.99 |
| all nine | fixed_two_stop | 211 | 0.711 | 0.905 | 0.393 | 25.25 | 73.53 |
| all nine | greedy_crossover | 35 | 0.971 | 1.000 | 0.657 | 40.86 | 252.02 |

### DEV BESIDE (agent - X, primary CI, verdict)

| row | dev20 (development, 19 races) | fresh five | earlier four | all nine |
| --- | --- | --- | --- | --- |
| agent - run_to_flag | -5.65 (-7.14, -4.23) ahead | +1.59 (-0.86, +4.05) level | -5.44 (-8.56, -2.98) ahead | -1.55 (-3.50, +0.33) level |
| agent - one_stop_total | -6.48 (-8.10, -4.97) ahead | +1.53 (-1.01, +4.08) level | -5.38 (-8.52, -2.92) ahead | -1.56 (-3.57, +0.36) level |
| agent - actual_strategy | -3.18 (-4.55, -1.90) ahead | +2.33 (-0.10, +4.64) level | -12.78 (-19.33, -7.29) ahead | -4.55 (-8.08, -1.38) ahead |
| agent - belief_optimizer | +0.14 (+0.02, +0.28) behind | -0.04 (-0.54, +0.36) level | +0.05 (-0.19, +0.28) level | -0.00 (-0.30, +0.24) level |
| agent mean regret s | 3.96 | 7.44 | 5.13 | 6.41 |
| agent share over 20 s | 0.043 | 0.194 | 0.040 | 0.125 |
| belief_optimizer share over 20 s | 0.043 | 0.202 | 0.040 | 0.129 |
| run_to_flag share over 20 s | 0.154 | 0.137 | 0.170 | 0.152 |
| one_stop_total share over 20 s | 0.175 | 0.137 | 0.160 | 0.147 |

## Agent

**Reliability:**
- 225 scenarios; 224 submitted and 1 not run (no_reference).
- Of the 224: 204 voluntary, 20 in a grace turn.
- 0 invalid, 0 errors, 0 API retries.
- 54 episodes hit the tool-call budget (201 calls refused).
- 42 turns gave no parseable tool call: replies cut at 1024 tokens, or JSON with unquoted keys.
  The loop recovered every one.

Grace submissions by race: France 5, Austria 2024 5, Australia 3, Belgium 3, Miami 2, Mexico 1,
Sakhir 1, Monaco 0, Austria 2025 0.

**Decisions:** 36 PIT, 188 STAY_OUT.

| track status | PIT | STAY_OUT |
| --- | --- | --- |
| green | 33 | 173 |
| SC | 1 | 13 |
| VSC | 2 | 2 |

**Tokens and calls per decision (pass):**
- input tokens: mean 84.5k, median 63.3k;
- output tokens: mean 943, median 797;
- tool calls: 17.8, of which 15.9 evidence calls and 4.2 repeats;
- in all: 18.9 M input and 0.21 M output tokens.

**Decision time:**

| run | episodes | latency mean / median / p90 s | tokens in / out | tool calls |
| --- | --- | --- | --- | --- |
| isolated (timing sample, 1 job) | 24 | 30.2 / 27.8 / 56.7 | 85.0k / 975 | 19.0 |
| concurrent (10 jobs), the same 24 | 24 | 55.2 / 57.4 / 96.0 | 97.7k / 999 | 19.9 |
| concurrent, all (as `agent_latency.csv` lists it: 225, 224 submitted) | 225 | 53.5 / 44.3 / 103.3 | 84.5k / 943 | 17.8 |

On the same 24 scenarios, concurrent minus isolated is +25.0 s (95% t interval +13.7 to +36.3), a
ratio of means of 1.83. The decision was the same on 23 of 24. For comparison, dev20 was isolated
32.9 / 27.6 / 49.4 s, and same decision 22/24.

**Lint:** 3 findings, explained above. Nothing else: no exception text, NaN, missing field or
identity leak, and no rule reply that fails to say what is legal.

**Scenarios under SC / VSC** (from scenarios.json): 207 green, 14 SC, 4 VSC.
- 2020 Sakhir: 3 SC.
- 2022 Miami: 6 SC, 2 VSC.
- 2022 Australia: 5 SC, 2 VSC.
- The other six races: all green.

**Belief's tyre-life prior tier, per race** (`tyre_life.prior_source` in the model reports in the
trajectories; tiers per docs/eval.md: 1 same circuit and compound, 2 same circuit, 3 pool only,
4 no usable earlier race):
- 2024 Austria and 2025 Austria: tier 1, every compound.
- 2019 Monaco: tier 4, the constant shares (the earliest cached season).
- 2020 Sakhir, 2021 France, 2021 Mexico, 2022 Miami, 2022 Australia, 2022 Belgium: tier 3, pool
  only, every compound reported.

So every fresh race priced its tyre life from the pool alone, with no history at the same circuit.
2022 Australia is the race the belief got most wrong.

## Ten trajectories read by eye

Each entry gives: the race and scenario; agent regret; belief_optimizer (bo) and run_to_flag (rtf)
regret; what the model did; and whether it helped or hurt.

1. **2021 France RAI:00** (lap 0, HARD; fresh). Agent 8.86, bo 9.17, rtf 4.49.
   - **Wrong:** eight times it asked `evaluate_strategy` for a pit with `stop_lap: 3`.
     `stop_lap` is not in the tool's schema, so `_dispatch` silently dropped it. Each call
     evaluated a pit at the end of lap 1 and got the same minimum-stint reply.
   - The model then ran out of budget and grace-submitted STAY_OUT 3.
   - Effect: wasted the budget. Result-neutral here (slightly better than bo).
   - Across the pass, 2 episodes sent `stop_lap` (16 calls) and 1 sent `alternative` (6 calls),
     both dropped silently.
2. **2022 Miami VER:00** (lap 0, MEDIUM; fresh). Agent 27.58, bo 27.58, rtf 0.86.
   - It checked the stays systematically (3 to 23 laps) and ran compare_plans on 4.
   - One reply was cut at 1024 tokens; the loop recovered.
   - It submitted STAY_OUT 16, which resolves to MEDIUM 16 / SOFT 22 / SOFT 19. The world's
     optimum is MEDIUM 22 / HARD 35.
   - Effect: neutral against bo, hurts against the rules. The belief never favours the long HARD
     stint.
3. **2022 Australia BOT:10** (lap 10, MEDIUM, green; fresh; the worst agent row). Agent 28.46,
   bo 30.66, rtf 11.71.
   - Careful comparisons, with repeats; then STAY_OUT 3 resolving to MEDIUM 15 / MEDIUM 25 /
     SOFT 8.
   - The world's optimum is MEDIUM 5 / HARD 43. No HARD long-stint candidate was offered by the
     tools.
   - Effect: slightly better than bo, much worse than rtf.
4. **2022 Australia ZHO:40** (under VSC, HARD age 19, 18 to go; fresh). Agent 19.47, bo 19.47,
   rtf 0.00.
   - The tools priced a stop now at 12.4 s under VSC and showed PIT MEDIUM faster in 72% of draws.
     The model pitted, as bo did. The world says run HARD to the flag.
   - **Wrong:** the rationale says the HARD set is "nearing its longest observed run (40 laps)"
     at age 19.
   - Effect: neutral against bo, hurts against rtf.
5. **2022 Belgium MSC:05** (lap 5, MEDIUM; fresh). Agent 24.24, bo 11.46, rtf 22.05.
   - **The language model hurt here.** At step 6, compare_plans ranked MEDIUM 9 / MEDIUM 17 /
     SOFT 13 best (p_best 0.42).
   - At step 8 it compared a different set of four, all later stops plus the one-stop
     MEDIUM 21 / SOFT 18, and picked the one-stop on p_best 0.40 within that set. It never
     compared it head to head with the step-6 winner.
   - It submitted STAY_OUT 17 after hitting the budget.
   - About 12.8 s worse than bo. Belgium overall is still the agent's best fresh race against the
     rules.
6. **2021 Mexico GIO:00** (lap 0, MEDIUM; fresh). Agent 17.59, bo 14.56, rtf 1.56.
   - compare_plans gave p_best 0.52 to the one-stop MEDIUM 27 / HARD 44 (STAY_OUT 10) and 0.02 to
     MEDIUM 13 / HARD 30 / HARD 28 (STAY_OUT 7).
   - **Wrong:** it submitted STAY_OUT 7, stating "highest probability of being the best plan
     (p_best = 0.56)". It misread the tool output.
   - Effect: hurt by about 3 s against bo. The one-stop it set aside is close to the world's
     optimum (MEDIUM 33 / HARD 38), so the loss against the right reading is likely larger. It
     was not scored, so I don't have a number.
7. **2019 Monaco RIC:70** (late race, 8 to go, MEDIUM age 59; earlier four). Agent 18.84,
   bo 18.84, rtf 0.00.
   - PIT SOFT against stay was a coin flip (54/46, gap 0.1 s); it stayed.
   - **Wrong:** the rationale states the gap with the wrong sign ("-0.3 s in favour of staying
     out").
   - Effect: neutral against bo; the belief's mean also chose to stay. The world wanted the stop.
8. **2020 Sakhir OCO:65** (under SC, HARD age 24, 22 to go; earlier four). Agent 2.09, bo 2.09,
   rtf 2.61.
   - PIT SOFT under the SC (pit loss now 12.4 s), as bo did.
   - **Wrong:** the rationale says the HARD set is "past its longest observed run (35 laps)" at
     age 24. A misstatement, but the decision is sound.
   - Effect: neutral; slightly better than rtf.
9. **2024 Austria GAS:00** (lap 0, MEDIUM; earlier four). Agent 22.64, bo 22.64, rtf 16.33.
   - After four evaluations, 7 consecutive replies failed to parse (the same JSON error at
     column 116). It grace-submitted STAY_OUT 3, resolving to bo's plan MEDIUM 7 / HARD 31 /
     HARD 33.
   - Effect: wasted turns, result-neutral against bo. The belief's three-stint plan is worse than
     rtf here.
10. **2025 Austria PIA:45** (HARD age 21, 25 to go; earlier four). Agent 0.00, bo 0.09, rtf 22.49.
    - It evaluated every pit and stay and ran compare_plans (PIT SOFT p_best 0.66), then pitted
      for SOFT. The rationale is accurate.
    - Effect: helped. This is the kind of case behind the earlier four's lead over the rules.

**Patterns:**
- The language model mostly passes the belief's plan through. It equals bo's regret on 106 of 224
  and is level with it in every group.
- Its own errors cost in some episodes and saved a little in others:
  - Hurt: the Belgium set-switching and the Mexico misread p_best (cases 5 and 6).
  - Wrong rationales with no decision change: cases 4, 7 and 8.
  - Wasted budget: the silently dropped `stop_lap` argument and repeated identical parse failures
    (cases 1 and 9).
- The fresh-five shortfall against the rules is the belief's: tier-3 tyre-life priors with no
  same-circuit history, and two-stop plans where the world one-stops on HARD, in 2022 Australia
  and Miami.
- Not fixed, not rerun: reported as limitations.
