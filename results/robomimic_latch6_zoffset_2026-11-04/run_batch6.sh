#!/usr/bin/env bash
# Phase 6 matched batch: IDENTICAL condition/seed/schedule matrix to Phase 5
# (run_batch.sh in results/robomimic_latch_2026-11-04), but using latch6_rollout.py
# with the single intervention --latch-z-offset 0.015 (world-z lift of the held
# goal once at acquisition, frozen thereafter). Output is a fresh Phase 6 raw dir.
set -euo pipefail
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
OFF=0.015
RUN=/home/ubuntu/robomimic-native-phase6-20261104/matched_batch
mkdir -p "$RUN"
for seed in 0 1 2; do
  for condition in always-hold always-enable; do
    name=${condition}_seed${seed}
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition "$condition" --latch-z-offset "$OFF" --out "$RUN/$name" > "$RUN/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name complete"
  done
  case "$seed" in 0) closure=40;; 1) closure=32;; 2) closure=34;; esac
  for phase in approach:15 closure:$closure carrying:70; do
    label=${phase%%:*}; start=${phase##*:}
    for duration in 10 60; do
      name=${label}_${duration}_seed${seed}
      "$PYTHON" "$SCRIPT" --seed "$seed" --condition pause --pause-start "$start" --pause-steps "$duration" --latch-z-offset "$OFF" --out "$RUN/$name" > "$RUN/$name.log" 2>&1
      echo "[$(date +%H:%M:%S)] $name complete"
    done
  done
done
echo "[$(date +%H:%M:%S)] PHASE6_BATCH_COMPLETE runs=$(ls "$RUN" | grep -v '\.log$' | wc -l)"
