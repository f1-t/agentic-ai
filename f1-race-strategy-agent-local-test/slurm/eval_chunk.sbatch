#!/bin/bash
# One array task per chunk of scenario items for the optimizer and the baselines. Anything
# that calls an LLM stays on the desktop; this runs only the simulator side.
#   python -m src.sim.jobs manifest eval_chunk --profile sol --n-items <N_ITEMS> --manifest $OUT/manifest.json --out $OUT
#   N=$(python -c "import json;print(len(json.load(open('$OUT/manifest.json'))))")
#   sbatch --array=0-$((N-1))%50 slurm/eval_chunk.sbatch
# Needs an eval runner installed with jobs.set_eval_runner (the eval harness registers it).
# Partition, QOS, time limit and module names below are placeholders: verify them against
# ASU's Sol documentation before the first submission.
#SBATCH --job-name=sim-eval-chunk
#SBATCH --partition=general        # placeholder, check Sol docs
#SBATCH --qos=public               # placeholder, check Sol docs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
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

python -m src.sim.jobs eval_chunk --profile "$SIM_PROFILE" --manifest "$MANIFEST" \
    --task-id "$SLURM_ARRAY_TASK_ID" --out "$OUT"
