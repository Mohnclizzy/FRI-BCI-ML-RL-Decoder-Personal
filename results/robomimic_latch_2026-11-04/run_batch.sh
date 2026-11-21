#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch_2026-11-04/latch_rollout.py
RUN=/home/ubuntu/robomimic-native-phase5-20261104/matched_batch
mkdir -p "$RUN"
for seed in 0 1 2; do
  for condition in always-hold always-enable; do
    name=${condition}_seed${seed}
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition "$condition" --out "$RUN/$name" > "$RUN/$name.log" 2>&1
    echo "$name complete"
  done
  case "$seed" in 0) closure=40;; 1) closure=32;; 2) closure=34;; esac
  for phase in approach:15 closure:$closure carrying:70; do
    label=${phase%%:*}; start=${phase##*:}
    for duration in 10 60; do
      name=${label}_${duration}_seed${seed}
      "$PYTHON" "$SCRIPT" --seed "$seed" --condition pause --pause-start "$start" --pause-steps "$duration" --out "$RUN/$name" > "$RUN/$name.log" 2>&1
      echo "$name complete"
    done
  done
done
