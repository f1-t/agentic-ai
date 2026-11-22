#!/bin/bash
# Launches the test9 agent stage once. Usage: agent_launch.sh [--resume]
OUT=<out-dir>
cd <worktree at eval-v2-test>/f1-race-strategy-agent-local-test || exit 2
set -a && . <main checkout>/f1-race-strategy-agent-local-test/.env && set +a
export LLM_BACKEND=voyager
for v in AGENT_TURN_STATUS AGENT_REPEAT_MODE AGENT_SUBMIT_GRACE_STEPS VOYAGER_TEMPERATURE; do
  if [ -n "${!v+x}" ]; then echo "ABORT: $v is set"; exit 3; fi
done
export AGENT_TURN_REPEAT=run
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
s=$(date +%s)
echo "START $(date -u +%FT%TZ) args: $*"
<main checkout>/f1-race-strategy-agent-local-test/.venv/bin/python -m src.eval.rerun agent \
  --set test9 --out-dir "$OUT" --world-config default --belief-config default \
  --timing-sample 24 --jobs 10 "$@"
rc=$?
echo "END $(date -u +%FT%TZ) exit $rc WALL $(( $(date +%s) - s )) s"
exit $rc
