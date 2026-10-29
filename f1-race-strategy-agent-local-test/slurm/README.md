# Slurm templates for the batch jobs

These are templates. Nothing here has been submitted.

**Before the first submission, verify Sol's scheduler, partitions, QOS names, time limits and
module names against ASU's Sol documentation.** Slurm is the expectation, not something
confirmed from here. The `#SBATCH --partition`, `--qos` and `module load` lines are
placeholders.

## Three commands

Run from `f1-race-strategy-agent-local-test/` with the venv active and `SIM_PROFILE=sol`
(the templates set it). `OUT` is on scratch.

```bash
OUT=$SCRATCH/sim_out

# 1. manifest: the list of units; the array size is its length
python -m src.sim.jobs manifest g_race --profile sol --races 2021_04_spain --manifest $OUT/manifest.json --out $OUT
N=$(python -c "import json;print(len(json.load(open('$OUT/manifest.json'))))")

# 2. one array task per unit (K = how many run at once)
sbatch --array=0-$((N-1))%50 slurm/g_race.sbatch

# 3. aggregate whatever finished; safe to re-run, lists missing tasks
python -m src.sim.jobs aggregate g_race --profile sol --manifest $OUT/manifest.json --out $OUT
```

Tasks are idempotent. Resubmitting the array skips every task whose output exists with a meta
file matching the sim code version and the profile hash, so a failed or preempted array is
fixed by submitting it again.

| job | template | unit | output |
| --- | --- | --- | --- |
| `g_race` | `g_race.sbatch` | race id | `$OUT/g/<race_id>.parquet` |
| `h_fold` | `h_fold.sbatch` | held-out race id or `season:YYYY` | `$OUT/h/<calibrator>/<race_id>.parquet` and `.rmse.json` |
| `eval_chunk` | `eval_chunk.sbatch` | `chunk_NNNN` | `$OUT/eval/<chunk>.parquet` |

`build_weekend` has no template: the bulk build needs network and the F1 TV login, so it never
runs on Sol.

## What to ask Sol for

- Python 3.11 or newer with `requirements.txt` installed (fastf1 is needed only for the loader,
  not for network).
- CPU partition, 1 core and 2 GB per task, array jobs of up to a few hundred tasks.
- About 1 GB of scratch for frames and outputs.
- No inbound or outbound network on compute nodes: the frames are committed, and wandb runs
  offline (`WANDB_MODE=offline`; run `wandb sync` from a login node afterwards).
- Optional GPU partition for the MLP ensemble; not required. Set `h.mlp.device` to `cuda` in a
  copy of the profile and request a GPU in `h_fold.sbatch`.

The LLM calls of the eval stay on the desktop with the API key.
