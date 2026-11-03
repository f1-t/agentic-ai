# Runbook: the real calibration runs

From "Sarah's PR merged" to "report regenerated". One command runs every stage (SPEC 0021,
`src/sim/real_run.py`):

```bash
python -m src.sim.calibrate real-run --profile local          # desktop: everything in-process
python -m src.sim.calibrate real-run --profile sol            # Sol: emits Slurm arrays, submits nothing
```

Run every command here from `f1-race-strategy-agent-local-test/` with the venv active.

## What it runs

| stage | what | runs when | output |
| --- | --- | --- | --- |
| `eligibility` | `calibration.eligibility` per race | at least one race | `real_run/eligibility.json`, and the table printed |
| `cross_race` | the cross-race pass and leave-one-race-out per arm (SPEC 0016) | at least one eligible race | `cross_race/` (`run.json`, parquets) |
| `event_rates` | SC and VSC rates by circuit and era (SPEC 0011b) | `data/derived/neutralised_periods.csv` exists | `event_rates/` |
| `fuel_prior` | the per-circuit race-pace prior (SPEC 0020) | at least 8 eligible races | `fuel_prior/` |
| `references` | `python -m src.eval reference --check`, then each stale reference rebuilt with its own inputs | `data/references/` exists | `data/references/<set>/<key>.parquet` |
| `report` | the cross-race report | the cross-race pass finished | `cross_race/summary.md`, `factors.csv`, ... |

Paths are under `reports/calibration/` unless `--out` says otherwise. `--races auto` (the
default) takes every race in `cache/`, and `--races 2021_04_spain ...` takes the races you
name. A stage that cannot run is skipped, and the reason is printed. It is also written to
`reports/calibration/real_run/manifest.json`. The manifest has one entry per stage with its
status (`ran`, `current`, `skipped`, `pending` or `failed`), reason, inputs and whether each
input is present, outputs, code version (`sim_code_version` and git commit) and wall time.

Rerunning is safe. A stage whose inputs, code and profile are unchanged since it last ran,
and whose outputs are still there, is reported `current` and is not rerun. The cross-race pass
also resumes unit by unit: finished races and folds are skipped. If a run is interrupted, run
the same command again.

## Checklist

1. **Pull.** Run `git checkout main && git pull`. Reinstall with `pip install -r
   requirements.txt` if the requirements changed.
2. **Check the cache.**
   - `ls cache/*/` should show the new weekends.
   - `python -c "from src.sim.real_run import cached_races; print(cached_races())"` should
     list every race you expect, as `<season>_<round>_<country>` ids. A note means an event
     directory the schedule does not name.
   - Check whether `data/derived/neutralised_periods.csv` and `data/race_features.csv` arrived
     with the PR. Without the first, the event rates are skipped. Without the second, the
     reports have no circuit covariates.
3. **Run.**
   - **Local:** `python -m src.sim.calibrate real-run --profile local`. The cross-race units use
     the profile's `parallel.n_jobs` workers.
   - **Sol:** run `python -m src.sim.calibrate real-run --profile sol` inside an interactive
     allocation (`salloc`), not on the login node. The eligibility, event-rate, fuel-prior and
     reference stages run in that process. The per-race and per-fold units of the cross-race
     pass are written as array scripts under `reports/calibration/real_run/slurm/`. They are
     built from `slurm/g_race.sbatch` and `slurm/h_fold.sbatch`, so first check the
     placeholders in those templates (`slurm/README.md`). The command prints the `sbatch` lines
     and submits nothing. Submit them (`mkdir -p logs` first) and wait for them to finish
     (`squeue -u $USER`). Then run the same `real-run` command again. Each rerun skips the
     finished units and emits the next phase: the races and arms, then the prior-sd fix, then
     the tau grid. The cross-race stage reads `ran` and the report is written once nothing is
     left. Units that failed are emitted again on the next rerun.
   - Afterwards, run `git checkout -- cache/fastf1_http_cache.sqlite`, since loading touches
     it. Never commit it.
4. **Inspect the eligibility table** before anything else. It is printed first and saved to
   `real_run/eligibility.json`, and it is the first section of `cross_race/summary.md`.
   - Check that every race you expected is eligible.
   - For each one that is not, read its reasons: wet laps, a red flag, too few finishers, too
     few dry compounds or clean laps.
   - A race that is out for a data reason (a missing session, missing race control messages)
     goes back to the data team. The eligibility rule does not change to let it in.
5. **Then read the summary**, `reports/calibration/cross_race/summary.md`. Look at these first:
   - **The Current defaults block.** Each window's incumbent arm, its config hash and
     `matches the config`. Every row should say yes. A no means the report and the committed
     config disagree; stop and find out why before reading further.
   - **Whether anything passed.** Look at the `passes` column of each window's selection
     table, and at the `Decision for <window>` lines (real-run prints them). With fewer than 8
     eligible races the report is PROVISIONAL, and nothing is adopted, whatever passed.
   - After that: the prior-sd fix and the tau grid, coverage, the gate and pairs, and
     "What changed from Spain". Then `event_rates/summary.md` ("Chosen (forward)") and
     `fuel_prior/summary.md` (its Decision line, and the Sentence when the run is real).
6. **Check the manifest.** In `reports/calibration/real_run/manifest.json`, `complete` should
   be true. Every stage should be `ran` or `current`, or `skipped` with a reason you expected.
   The `references` entry lists each stale reference and whether it was rebuilt. A reference it
   could not rebuild (its scenario set or world config is gone) gives the reason.
   `python -m src.eval reference --check` still lists the superseded files. Remove them in the
   same PR as the rebuilt ones.
7. **Commit the regenerated reports** on a branch, through a PR:
   - `cross_race/` (its `.gitignore` leaves the parquets out), `event_rates/`, `fuel_prior/`,
     `real_run/manifest.json` and `real_run/eligibility.json`.
   - Any rebuilt references.
   - Not `real_run/slurm/`: the scripts carry absolute paths and are regenerated.

## What not to do

- **No tuning after seeing held-out numbers (plan section 18).** The arms, the selection rules
  and margins (`NONINFERIORITY_S`, `MIN_RACES`, the horizons), the tau multipliers, the
  eligibility rule, the candidate lists and the priors were all fixed before any cross-race
  data. Once you have seen the held-out tables, do not change any of them. Do not drop or add
  races to move a decision, and do not rerun until something passes. A change to a rule is a
  new spec, written down before it is run, and its evidence comes from races it has not seen.
- Do not adopt a PROVISIONAL choice, and do not edit `configs/sim/belief.yaml`, `world.yaml` or
  the `SimConfig` defaults from the report in the same PR. Adopting a choice is its own
  reviewed change.
- Do not submit `real_run/slurm/*.sbatch` before checking the template placeholders. Do not
  hand-edit the scripts either; rerun `real-run` and it writes them again.
- Do not run synthetic races into `reports/calibration/`. The stages refuse to, and a pipeline
  test goes to `--out` in a tmp directory.
- Do not fetch data on Sol: the compute nodes have no network, and the cache build needs the
  F1 TV login.
