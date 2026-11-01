#!/usr/bin/env bash
#SBATCH --job-name=fri-bci-train
# Supply account, partition, GPU resources and time using site-specific sbatch flags.
set -euo pipefail
: "${FRI_REPO:?Set absolute repository path}"
: "${ISAACLAB_ROOT:?Set pinned Isaac Lab checkout path}"
: "${RUN_DIR:?Set a new absolute output directory}"
cd "$FRI_REPO"
"$ISAACLAB_ROOT/.venv/bin/python" -m fri_bci_sim.cli train \
  --isaaclab-root "$ISAACLAB_ROOT" \
  --num-envs "${NUM_ENVS:-1024}" --iterations "${ITERATIONS:-1000}" \
  --seed "${SEED:-0}" --run-dir "$RUN_DIR" --execute
