#!/usr/bin/env bash
# Reproduction script for the two bounded follow-up diagnostics.
# Set RUN_ROOT to a fresh directory; existing run directories are never overwritten.
set -euo pipefail
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_continuous_2026-11-04/continuous_rollout.py
RUN_ROOT=${RUN_ROOT:-/home/ubuntu/robomimic-native-phase4-reproduction}
mkdir -p "$RUN_ROOT/closure_corrective" "$RUN_ROOT/resume_reset_diagnostic"
for pair in 0:40 1:32 2:34; do
  seed=${pair%%:*}; start=${pair##*:}
  for duration in 10 60; do
    name=closure_at_${start}_${duration}_seed${seed}
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition pause --pause-start "$start" --pause-steps "$duration" --out "$RUN_ROOT/closure_corrective/$name" > "$RUN_ROOT/closure_corrective/$name.log" 2>&1
  done
  for phase in closure carrying; do
    if [ "$phase" = carrying ]; then start_at=70; else start_at=$start; fi
    name=${phase}_${start_at}_10_seed${seed}
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition pause --pause-start "$start_at" --pause-steps 10 --resume-state reset --out "$RUN_ROOT/resume_reset_diagnostic/$name" > "$RUN_ROOT/resume_reset_diagnostic/$name.log" 2>&1
  done
done
