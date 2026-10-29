#!/bin/bash
# One array task per fold (held-out race, or season:YYYY when the profile has loso on).
#   python -m src.sim.jobs manifest h_fold --profile sol --races <ids...> --manifest $OUT/manifest.json --out $OUT
#   N=$(python -c "import json;print(len(json.load(open('$OUT/manifest.json'))))")
#   sbatch --array=0-$((N-1))%50 --export=ALL,SIM_RACES="<ids...>",SIM_CALIBRATOR=<name> slurm/h_fold.sbatch
# Partition, QOS, time limit and module names below are placeholders: verify them against
# ASU's Sol documentation before the first submission. For the MLP on a GPU partition, set
# h.mlp.device to cuda in a copy of the profile and request a GPU here.
#SBATCH --job-name=sim-h-fold
#SBATCH --partition=general        # placeholder, check Sol docs
#SBATCH --qos=public               # placeholder, check Sol docs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load python/3.11            # placeholder module name
source "${SIM_VENV:-$HOME/venvs/f1-sim}/bin/activate"

export SIM_PROFILE=sol
export WANDB_MODE=offline
export OMP_NUM_THREADS=1

cd "${SIM_REPO:-$HOME/agentic-ai/f1-race-strategy-agent-local-test}"
OUT="${SIM_OUT:-$SCRATCH/sim_out}"
MANIFEST="${SIM_MANIFEST:-$OUT/manifest.json}"
mkdir -p logs "$OUT"

# SIM_RACES is the training pool (space separated race ids); unquoted on purpose.
python -m src.sim.jobs h_fold --profile "$SIM_PROFILE" --manifest "$MANIFEST" \
    --task-id "$SLURM_ARRAY_TASK_ID" --out "$OUT" \
    --calibrator "${SIM_CALIBRATOR:?set SIM_CALIBRATOR}" --pool ${SIM_RACES:?set SIM_RACES}
