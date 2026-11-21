#!/usr/bin/env bash
# S006R12 recorded-EEG learned-policy replay (authorized bounded exploratory run).
# Drives the native BC-RNN (Phase 2 checkpoint) with the recorded S006R12 decision
# trace via the Phase 6 harness (latch6_rollout.py --condition eeg): always-enable /
# HOLD switching driven by the causal decision trace, completion latch (+0.015 m z)
# established in Phase 6, causal timestamps, native 400-step (20 s @ 20 Hz) budget.
# Matched matrix: 3 gates x seeds 0-2. Skips any output dir that already exists.
set -uo pipefail
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/ubuntu/robomimic-native-phase2-20261104/src/robosuite:./simulation/src
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
DEC=outputs/isaac/eeg-batch-inputs/S006R12-decisions.jsonl
OFF=0.015
BASE=results/robomimic_eeg_s006r12_2026-11-04
for gate in raw confidence temporal; do
  for seed in 0 1 2; do
    name=${gate}_seed${seed}
    OUT=$BASE/$name
    if [ -d "$OUT" ]; then echo "[skip] $name (dir exists)"; continue; fi
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition eeg --gate "$gate" \
      --decisions "$DEC" --latch-z-offset "$OFF" --out "$OUT" > "$BASE/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name exit=$?"
  done
done
echo "EEG_BATCH_COMPLETE runs=$(ls "$BASE" | grep -v '\.log$' | grep -v '\.sh$' | wc -l)"
