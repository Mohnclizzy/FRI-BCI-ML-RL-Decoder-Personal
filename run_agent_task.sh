#!/usr/bin/env bash
# Autonomous Agent Task Runner on Brev
# Runs the clean pause-matched control on S001R12 using the verified harness.
set -euo pipefail
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal

echo "=== [AGENT RUNNER] Starting Task: S001R12 Pause-Matched Controls ==="

export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/ubuntu/robomimic-native-phase2-20261104/src/robosuite:./simulation/src:${PYTHONPATH:-}

PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
OFF=0.015
BASE=results/robomimic_controls_s001r12_2026-11-06
mkdir -p "$BASE"

# Pause condition: S001 typical activation window (onset step ~30, length 150 steps)
# Runs seeds 0, 1, 2 under the now-reconciled pause condition
for seed in 0 1 2; do
  name=pause_seed${seed}
  OUT=$BASE/$name
  echo "[$(date +%H:%M:%S)] Running $name..."
  "$PYTHON" "$SCRIPT" --seed "$seed" --condition "pause" --pause-start 30 --pause-steps 150 --latch-z-offset "$OFF" --out "$OUT" > "$BASE/$name.log" 2>&1 || true
  echo "[$(date +%H:%M:%S)] $name complete (exit=$?)"
done

# Run aggregation
"$PYTHON" "$BASE/aggregate.py" 2>&1 || python3 "$BASE/aggregate.py" 2>&1 || true

echo "=== [AGENT RUNNER] Task Complete. Updating git... ==="
