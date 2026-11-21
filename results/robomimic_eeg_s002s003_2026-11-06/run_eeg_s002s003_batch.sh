#!/usr/bin/env bash
# S002R12 + S003R12 HELD-OUT recorded-EEG learned-policy replay (subjects S002, S003).
#
# S002 / S003 are HELD-OUT runs: each model was calibrated on its OWN subject's
# runs 4/8 (per-person CSP/LDA), with run 12 held out and decoded offline. No live
# EEG, no new robot, no Franka: native robomimic/MuJoCo only.
#
# Identical harness + regime to the S001R12 held-out and S006R12 training-subject replays:
#   latch6_rollout.py --condition eeg
#   - Phase 2 BC-RNN checkpoint (lift_ph_low_dim_epoch_1000_succ_100.pth)
#   - Phase 6 completion latch (+0.015 m world-z at acquisition, established in Phase 6)
#   - always-hold / policy switching driven CAUSALLY by the per-subject decision trace
#   - native 400-step (20 s @ 20 Hz) budget, CPU/MuJoCo
# Matched matrix: 2 subjects x 3 gates (raw/confidence/temporal) x seeds 0-2 = 18 runs.
# Skips any output dir that already exists. The pause-control harness bug (lines 85-86)
# is NOT exercised here: this batch is --condition eeg only (see addendum 2026-11-06).
set -uo pipefail
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/ubuntu/robomimic-native-phase2-20261104/src/robosuite:./simulation/src
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
OFF=0.015
BASE=results/robomimic_eeg_s002s003_2026-11-06
n=0
for S in S002 S003; do
  DEC=$BASE/$S/${S}R12_decisions.jsonl
  for gate in raw confidence temporal; do
    for seed in 0 1 2; do
      name=${S}_${gate}_seed${seed}
      OUT=$BASE/$name
      if [ -d "$OUT" ]; then echo "[skip] $name (dir exists)"; continue; fi
      "$PYTHON" "$SCRIPT" --seed "$seed" --condition eeg --gate "$gate" \
        --decisions "$DEC" --latch-z-offset "$OFF" --out "$OUT" > "$BASE/$name.log" 2>&1
      rc=$?
      echo "[$(date +%H:%M:%S)] $name exit=$rc"
      n=$((n+1))
    done
  done
done
echo "S002S003_EEG_BATCH_COMPLETE runs=$n"
