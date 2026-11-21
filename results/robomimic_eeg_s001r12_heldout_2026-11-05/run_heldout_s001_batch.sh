#!/usr/bin/env bash
# S001R12 HELD-OUT recorded-EEG learned-policy replay (held-out qualification, subject S001).
#
# S001 is a HELD-OUT subject (not in the Phase 2 / 5-person training set S006-S010).
# Its run-12 decision trace was decoded offline with the frozen per-person CSP/LDA
# calibration (S001's own runs 4/8), with run 12 held out. The trace is pre-decoded;
# no live EEG, no new robot, no Franka: native robosuite/MuJoCo only.
#
# Identical harness + regime to the S006R12 training-subject replay:
#   latch6_rollout.py --condition eeg
#   - Phase 2 BC-RNN checkpoint (lift_ph_low_dim_epoch_1000_succ_100.pth)
#   - Phase 6 completion latch (+0.015 m z, established in Phase 6)
#   - always-enable / HOLD switching driven causally by the decision trace
#   - native 400-step (20 s @ 20 Hz) budget, CPU/MuJoCo
# Matched matrix: 3 gates x seeds 0-2 (9 runs). Skips any output dir that exists.
set -uo pipefail
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/ubuntu/robomimic-native-phase2-20261104/src/robosuite:./simulation/src
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
DEC=/home/ubuntu/S001R12-decisions.jsonl
OFF=0.015
BASE=results/robomimic_eeg_s001r12_heldout_2026-11-05
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
echo "HELDOUT_S001_BATCH_COMPLETE runs=$(ls "$BASE" | grep -v '\.log$' | grep -v '\.sh$' | wc -l)"
