#!/usr/bin/env bash
# Verified v2 Robot Control & Matched-Schedule Runner (robomimic-latch6-v2.0.0)
# Uses explicit enable-window, hold-window, and enable-schedule semantics from
# simulation/src/fri_bci_sim/robomimic_harness.py. Ambiguous `--condition pause`
# and error-masking `|| true` are prohibited.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== [AGENT RUNNER] Starting Verified v2 Matched-Control Rollouts (robomimic-latch6-v2.0.0) ==="

ROBOMIMIC_RUNTIME_ROOT="${ROBOMIMIC_RUNTIME_ROOT:-/home/ubuntu/robomimic-native-phase2-20261104}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$ROBOMIMIC_RUNTIME_ROOT/runtime/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$REPO_ROOT/simulation/src:$REPO_ROOT:${ROBOMIMIC_RUNTIME_ROOT}/src/robosuite:${PYTHONPATH:-}"

PYTHON="${PYTHON:-$ROBOMIMIC_RUNTIME_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python interpreter not found or not executable at: $PYTHON" >&2
  exit 1
fi

# Preflight verification: fail fast before running rollouts if runtime dependencies are missing
"$PYTHON" -c "import fri_bci_sim.robomimic_harness as h; assert h.HARNESS_VERSION == 'robomimic-latch6-v2.0.0'"
"$PYTHON" -c "import robomimic, robosuite, mujoco_py"

OFF=0.015
BASE="${OUT_BASE:-results/robomimic_controls_v2_2026-11-11}"
SCHEDULES_JSON="results/project_audit_2026-11-11/matched_control_schedules_20s.json"
mkdir -p "$BASE"

# 1. Explicit enable-window vs hold-window controls on S001 (window_start=30, window_steps=150)
for cond in enable-window hold-window; do
  for seed in 0 1 2; do
    name="${cond}_start30_steps150_seed${seed}"
    OUT="$BASE/$name"
    echo "[$(date +%H:%M:%S)] Running $name..."
    "$PYTHON" -m fri_bci_sim.robomimic_harness \
      --seed "$seed" \
      --condition "$cond" \
      --window-start 30 \
      --window-steps 150 \
      --latch-z-offset "$OFF" \
      --out "$OUT" > "$BASE/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name complete"
  done
done

# 2. Diagnostic 20s-horizon boundary enable-window sweep (36, 48, 60, 80, 110, 130, 212, 218 steps)
for steps in 36 48 60 80 110 130 212 218; do
  for seed in 0 1 2; do
    name="enable-window_start1_steps${steps}_seed${seed}"
    OUT="$BASE/$name"
    echo "[$(date +%H:%M:%S)] Running $name..."
    "$PYTHON" -m fri_bci_sim.robomimic_harness \
      --seed "$seed" \
      --condition "enable-window" \
      --window-start 1 \
      --window-steps "$steps" \
      --latch-z-offset "$OFF" \
      --out "$OUT" > "$BASE/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name complete"
  done
done

echo "=== [AGENT RUNNER] All v2 matched-control rollouts completed cleanly ==="
