#!/bin/bash
# One array task per race. Submit with the manifest length N:
#   N=$(python -c "import json;print(len(json.load(open('manifest.json'))))")
#   sbatch --array=0-$((N-1))%50 slurm/g_race.sbatch
# Partition, QOS, time limit and module names below are placeholders: verify them against
# ASU's Sol documentation before the first submission.
#SBATCH --job-name=sim-g-race
#SBATCH --partition=general        # placeholder, check Sol docs
#SBATCH --qos=public               # placeholder, check Sol docs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# --array is passed on the sbatch command line (0-$(N-1)%K, N = manifest length, K = concurrent tasks)

set -euo pipefail

# --- environment: module / venv activation (placeholders, check Sol docs) -----------
module load python/3.11            # placeholder module name
source "${SIM_VENV:-$HOME/venvs/f1-sim}/bin/activate"

export SIM_PROFILE=sol
export WANDB_MODE=offline          # no network on compute nodes; sync afterwards with `wandb sync`
export OMP_NUM_THREADS=1

cd "${SIM_REPO:-$HOME/agentic-ai/f1-race-strategy-agent-local-test}"
OUT="${SIM_OUT:-$SCRATCH/sim_out}"
MANIFEST="${SIM_MANIFEST:-$OUT/manifest.json}"
mkdir -p logs "$OUT"

python -m src.sim.jobs g_race --profile "$SIM_PROFILE" --manifest "$MANIFEST" \
    --task-id "$SLURM_ARRAY_TASK_ID" --out "$OUT"
